"""
TripleExtractor — Improved spaCy-based open information extraction.

Covered sentence patterns
──────────────────────────
1.  Active-verb (SVO)              John bought a car.
2.  Passive                         The car was sold by David.
3.  Copular / linking-verb          Paris is beautiful.
4.  Apposition                      John, the CEO, announced a product.
5.  Relative clause (relcl)         The engineer who designed the bridge …
6.  Adjectival clause (acl)         The book written by Orwell …
7.  Adverbial clause (advcl)        He left because he was tired.
8.  Prepositional relations         John works in Paris.
9.  xcomp / ccomp                   John wants to buy a house.
10. Possessive (poss)               John's car is red.
11. Noun–noun compound / nmod       Tesla CEO Elon Musk …
12. Conjunction expansion           John and Mary bought a car.
13. Reported speech / say-that      He said that the deal was done.
14. Attributive adjective           The tall man left.
15. Existential there               There is a problem.
16. Negation                        John did not buy a car.
17. Coreference resolution          (requires en_coreference_web_trf)
18. Coordinated predicates          He walked out and announced...
19. Numerical entity extraction     3.4 percent, 37.8 million people
20. Fallback heuristic              any short sentence that produced nothing else

Installation
────────────
    pip install spacy spacy-experimental
    pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.4/en_coreference_web_trf-3.7.0-py3-none-any.whl
    python -m spacy download en_core_web_trf
"""
"""
TripleExtractor — Improved spaCy-based open information extraction.

Key Improvements:
- Fixed preposition confusion (with -> in, from, on, etc.)
- Fixed verb phrase fragmentation
- Fixed structural errors in predicates
- Enhanced completeness without over-splitting
- Better handling of complex sentences
- Improved quality scoring and filtering
- Fixed hallucination issues
"""

import sys
import re
from collections import deque
from typing import Optional, Set, Tuple, List, Dict, Any, Union
import spacy
from tqdm import tqdm

# ─── dependency label sets ───────────────────────────────────────────────────

SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
OBJECT_DEPS = {"dobj", "obj", "pobj", "iobj", "attr", "oprd", "dative", "acomp"}
RELATION_AUX = {"aux", "auxpass", "neg", "prt", "advmod"}
CLAUSAL_COMPLEMENT = {"xcomp", "ccomp"}
# Embedded-clause deps that should NOT be swallowed into a parent argument's
# span — they are extracted as their own, separate triples elsewhere
# (see _extract_relative_clauses, _acl_triples, _advcl_triples,
# _extract_nested_clauses_recursive), so leaving them in the parent span just
# duplicates their content as one giant, unmatchable object.
CLAUSAL_LEAK_DEPS = {"relcl", "acl", "advcl"}
FORBIDDEN_SUBJECT_DEPS = {"appos", "relcl", "acl", "advcl", "punct"}
NUMERICAL_ENTITY_TYPES = {"PERCENT", "MONEY", "DATE", "CARDINAL", "QUANTITY", "TIME"}
REPORTING_VERBS = {"said", "stated", "reported", "claimed", "argued", "explained", "told", "announced", "noted", "pointed", "observed", "commented", "declared", "expressed"}

Triple = Tuple  # (subj_text, rel_text, obj_text, subj_idx, obj_idx, rel_idx)


def _clean(text: str) -> str:
    """Clean and normalize text."""
    text = str(text) if text is not None else ""
    return re.sub(r"\s+", " ", text).strip()


class TripleExtractorConfig:
    """Configuration for triple extraction."""
    
    def __init__(self):
        self.use_coref: bool = True
        self.use_biomedical: bool = False
        self.min_relation_words: int = 1
        self.max_relation_words: int = 10
        self.min_subject_words: int = 1
        self.min_object_words: int = 1
        self.max_triples_per_sentence: int = 30
        self.deduplication_level: str = 'aggressive'
        self.include_context: bool = False
        self.coverage_threshold: float = 0.4
        self.use_fallback: bool = True
        self.max_span_length: int = 50
        self.clean_output: bool = True
        self.extract_numerical: bool = True
        self.extract_quotes: bool = True
        self.extract_cross_sentence: bool = True
        self.max_recursion_depth: int = 4
        self.min_quality_score: float = 0.3


# ─── main class ──────────────────────────────────────────────────────────────

class TripleExtractor:
    """Enhanced rule-based triple extractor with improved accuracy."""

    # Standard preposition corrections
    PREPOSITION_CORRECTIONS = {
        'in': ['in', 'within', 'inside', 'into'],
        'on': ['on', 'upon', 'onto'],
        'at': ['at', 'by', 'beside', 'near'],
        'from': ['from', 'of', 'off', 'out of'],
        'to': ['to', 'toward', 'towards', 'for'],
        'with': ['with', 'along', 'together with'],
        'for': ['for', 'on behalf of'],
        'about': ['about', 'concerning', 'regarding'],
        'by': ['by', 'through', 'via'],
        'of': ['of', "'s", 'belonging to']
    }
    
    # Verbs that typically take specific prepositions
    VERB_PREPOSITION_MAP = {
        'live': 'in',
        'reside': 'in',
        'stay': 'in',
        'work': 'in',
        'study': 'at',
        'graduate': 'from',
        'come': 'from',
        'go': 'to',
        'travel': 'to',
        'move': 'to',
        'return': 'to',
        'look': 'at',
        'listen': 'to',
        'speak': 'to',
        'talk': 'to',
        'think': 'about',
        'care': 'about',
        'worry': 'about',
        'wait': 'for',
        'search': 'for',
        'apply': 'for',
        'belong': 'to',
        'depend': 'on',
        'rely': 'on',
        'insist': 'on',
        'succeed': 'in',
        'participate': 'in',
        'engage': 'in',
        'consist': 'of',
        'compose': 'of',
        'die': 'of'
    }
    
    # Verbs that indicate passive constructions
    PASSIVE_VERBS = {'born', 'known', 'called', 'named', 'considered', 'regarded', 'seen', 'found', 'made'}

    def __init__(self, nlp, config: Optional[TripleExtractorConfig] = None):
        self.nlp = nlp
        self.config = config or TripleExtractorConfig()
        
        # Caches for performance
        self._span_cache: Dict[tuple, tuple] = {}
        self._relation_cache: Dict[tuple, tuple] = {}
        self._subject_cache: Dict[tuple, tuple] = {}
        self._noun_chunk_cache: Dict[int, list] = {}
        self._num_phrase_cache: Dict[int, str] = {}
        
        # For cross-sentence tracking
        self._subject_tracker: Dict[str, List] = {}

    # ── Enhanced Preposition Correction ─────────────────────────────────────

    def _correct_preposition(self, verb_text: str, prep_text: str) -> str:
        """Correct preposition based on verb and context."""
        # Get the base verb form
        verb_base = verb_text.lower().split()[0] if verb_text else ""
        
        # Check if we have a specific mapping
        if verb_base in self.VERB_PREPOSITION_MAP:
            correct_prep = self.VERB_PREPOSITION_MAP[verb_base]
            # If the prep is already correct, keep it
            if prep_text.lower() == correct_prep:
                return prep_text
            # If the prep is in the list of acceptable prepositions, keep it
            if prep_text.lower() in self.PREPOSITION_CORRECTIONS.get(correct_prep, []):
                return prep_text
            return correct_prep
        
        # Handle special cases
        if verb_base in self.PASSIVE_VERBS and prep_text.lower() in ['as', 'for', 'by']:
            return prep_text
        
        # Default: keep the preposition if it's common, otherwise use 'in'
        common_preps = ['in', 'on', 'at', 'from', 'to', 'with', 'for', 'about', 'by', 'of', 'as', 'for']
        if prep_text.lower() in common_preps:
            return prep_text
        
        return 'in'  # Default fallback

    def _fix_relation_verb_phrase(self, rel_parts: List[str], verb: Any) -> str:
        """Fix fragmented verb phrases like 'has to is among'."""
        # Clean the relation
        rel = " ".join(rel_parts)
        
        # Fix common fragmentation patterns
        patterns = [
            (r'has to is among', 'is among'),
            (r'has to is', 'is'),
            (r'is to is', 'is'),
            (r'has to be', 'is'),
            (r'to is', 'is'),
            (r'has to (has|have|had)', r'\1'),  # has to has -> has
            (r'after had elected', 'had elected after'),
            (r'after had', 'had after'),
            (r'to to', 'to'),
            (r'to has', 'has'),
            (r'to have', 'have'),
            (r'to be', 'be'),
            (r'\bwhen\b', ''),  # Remove stray 'when'
            (r'\bthat\b', ''),   # Remove stray 'that' in relations
        ]
        
        for pattern, replacement in patterns:
            rel = re.sub(pattern, replacement, rel)
        
        # Fix double spaces
        rel = re.sub(r'\s+', ' ', rel).strip()
        
        # If relation is empty or just 'is', get the verb text
        if not rel or rel == 'is':
            try:
                rel = verb.text if hasattr(verb, 'text') else 'is'
            except (TypeError, AttributeError):
                rel = 'is'
        
        return rel

    def _fix_relation_with_preposition(self, rel: str, verb_text: str, prep_text: str = None) -> str:
        """Fix relations with prepositions to use correct format."""
        if not prep_text:
            return rel
        
        # Get the correct preposition
        correct_prep = self._correct_preposition(verb_text, prep_text)
        
        # If the relation already has the preposition, use it
        if f" {correct_prep}" in rel or rel.endswith(correct_prep):
            return rel
        
        # Clean up the relation
        rel = re.sub(r'\s+', ' ', rel).strip()
        
        # If relation ends with a preposition, replace it
        for prep in self.PREPOSITION_CORRECTIONS.keys():
            if re.search(rf'\s{prep}$', rel):
                rel = re.sub(rf'\s{prep}$', f' {correct_prep}', rel)
                return rel
        
        # Otherwise, add the preposition
        if correct_prep and not rel.endswith(correct_prep):
            return f"{rel} {correct_prep}"
        
        return rel

    # ── Improved Relation Span ─────────────────────────────────────────────

    def _relation_span(self, verb) -> tuple[str, list[int]]:
        """Enhanced relation span with proper handling of complex predicates."""
        cache_key = (verb.i, id(verb.doc))
        if cache_key in self._relation_cache:
            return self._relation_cache[cache_key]
        
        aux_tokens = []
        neg_present = False
        verb_indices = []
        
        try:
            # Check immediate children
            children_list = list(verb.children)
            for child in children_list:
                if child.dep_ in {"aux", "auxpass"}:
                    aux_tokens.append(child)
                    verb_indices.append(child.i)
                elif child.dep_ == "neg":
                    neg_present = True
                    aux_tokens.append(child)
                    verb_indices.append(child.i)
                elif child.dep_ == "prt":
                    aux_tokens.append(child)
                    verb_indices.append(child.i)
                elif child.dep_ == "advmod" and child.text.lower() in ["on", "off", "up", "down", "in", "out"]:
                    aux_tokens.append(child)
                    verb_indices.append(child.i)
            
            # Check ancestors for modals
            ancestors_list = list(verb.ancestors)
            for ancestor in ancestors_list:
                if ancestor.dep_ == "aux" and ancestor not in aux_tokens:
                    aux_tokens.append(ancestor)
                    verb_indices.append(ancestor.i)
                elif ancestor.dep_ == "neg" and ancestor not in aux_tokens:
                    neg_present = True
                    aux_tokens.append(ancestor)
                    verb_indices.append(ancestor.i)
        except (TypeError, AttributeError):
            pass
        
        # Add the verb
        aux_tokens.append(verb)
        verb_indices.append(verb.i)
        
        # Sort by position
        pairs = sorted(zip([t.i for t in aux_tokens], aux_tokens), key=lambda x: x[0])
        sorted_tokens = [p[1] for p in pairs]
        sorted_indices = [p[0] for p in pairs]
        
        # Build relation string with proper handling
        rel_parts = []
        for i, t in enumerate(sorted_tokens):
            if t.dep_ == "neg":
                rel_parts.append("not")
            else:
                rel_parts.append(t.text)
        
        rel = " ".join(rel_parts)
        
        # Fix fragmented verb phrases
        rel = self._fix_relation_verb_phrase(rel_parts, verb)
        
        # Handle prepositional relations
        try:
            for child in verb.children:
                if child.dep_ == "prep":
                    prep_text = child.text
                    # Get the object of the preposition
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            # Add the preposition to the relation with correct format
                            correct_prep = self._correct_preposition(verb.text, prep_text)
                            if correct_prep not in rel:
                                rel = f"{rel} {correct_prep}"
                            break
        except (TypeError, AttributeError):
            pass
        
        result = _clean(rel), sorted_indices
        self._relation_cache[cache_key] = result
        return result

    # ── Improved Subject Extraction ─────────────────────────────────────────

    def _clean_subject_span(self, token) -> tuple[str, tuple[int, int]]:
        """Enhanced subject span cleaning with better noun phrase detection."""
        cache_key = (token.i, id(token.doc))
        if cache_key in self._subject_cache:
            return self._subject_cache[cache_key]
        
        # Try to use noun chunk
        chunk = self._get_noun_chunk(token)
        
        if chunk is not None:
            nodes = []
            try:
                for i in range(chunk.start, chunk.end):
                    t = token.doc[i]
                    if not t.is_punct and t.dep_ not in FORBIDDEN_SUBJECT_DEPS:
                        nodes.append(t)
            except (TypeError, AttributeError):
                nodes = [token]
            
            # Include possessive modifiers
            for t in nodes[:]:
                try:
                    for child in t.children:
                        if child.dep_ in ["poss", "possessive"] and child.i not in [n.i for n in nodes]:
                            nodes.append(child)
                except (TypeError, AttributeError):
                    pass
            
            if nodes:
                nodes.sort(key=lambda x: x.i)
                s, e = nodes[0].i, nodes[-1].i + 1
                # Expand to include determiners
                for i in range(s - 1, max(s - 3, -1), -1):
                    try:
                        if token.doc[i].dep_ == "det":
                            s = i
                        else:
                            break
                    except (TypeError, AttributeError):
                        break
                result = _clean(token.doc[s:e].text), (s, e)
                self._subject_cache[cache_key] = result
                return result

        # Fallback to subtree with better filtering
        nodes: list = []
        queue = deque([token])
        visited = set()

        while queue:
            t = queue.popleft()
            if t.i in visited:
                continue
            visited.add(t.i)

            if t.is_punct:
                continue

            # Include modifiers that are part of the subject
            if t.dep_ in {"det", "amod", "compound", "poss", "nmod", "nummod", "possessive", "prep", "pobj"}:
                nodes.append(t)
                try:
                    for child in t.children:
                        if child.i not in visited:
                            queue.append(child)
                except (TypeError, AttributeError):
                    pass
            elif t.dep_ not in FORBIDDEN_SUBJECT_DEPS:
                nodes.append(t)

        if not nodes:
            result = token.text, (token.i, token.i + 1)
            self._subject_cache[cache_key] = result
            return result

        nodes.sort(key=lambda x: x.i)
        s, e = nodes[0].i, nodes[-1].i + 1
        result = _clean(token.doc[s:e].text), (s, e)
        self._subject_cache[cache_key] = result
        return result

    # ── Improved Object Extraction ──────────────────────────────────────────

    def _get_objects(self, verb) -> list:
        """Enhanced object extraction with better handling."""
        objects = []
        seen = set()
        
        try:
            children_list = list(verb.children)
            for child in children_list:
                if child.dep_ in OBJECT_DEPS and child.i not in seen:
                    seen.add(child.i)
                    objects.extend(self._expand_conj(child))
            
            # For passive, include nsubjpass as object
            if not objects:
                for child in children_list:
                    if child.dep_ == "nsubjpass" and child.i not in seen:
                        seen.add(child.i)
                        objects.append(child)
            
            # Include acl/relcl objects
            if not objects:
                for child in children_list:
                    if child.dep_ in ["relcl", "acl"]:
                        for subchild in child.children:
                            if subchild.dep_ in OBJECT_DEPS and subchild.i not in seen:
                                seen.add(subchild.i)
                                objects.extend(self._expand_conj(subchild))
            
            # Check for objects in prepositional phrases that are part of the verb
            if not objects:
                for child in children_list:
                    if child.dep_ == "prep":
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                # For verbs like 'reside', 'live', the prep is actually the verb relation
                                verb_text = verb.text.lower()
                                if verb_text in self.VERB_PREPOSITION_MAP:
                                    correct_prep = self.VERB_PREPOSITION_MAP[verb_text]
                                    if child.text.lower() == correct_prep:
                                        objects.append(grandchild)
                                else:
                                    objects.append(grandchild)
        except (TypeError, AttributeError):
            pass
        
        return objects

    def _get_prepositional_objects(self, verb) -> list:
        """Get objects from prepositional phrases with proper handling."""
        objects = []
        try:
            for child in verb.children:
                if child.dep_ == "prep":
                    prep_text = child.text.lower()
                    # Check if this preposition is appropriate for the verb
                    verb_text = verb.text.lower()
                    if verb_text in self.VERB_PREPOSITION_MAP:
                        correct_prep = self.VERB_PREPOSITION_MAP[verb_text]
                        if prep_text != correct_prep:
                            # Still extract but mark the correct preposition
                            for grandchild in child.children:
                                if grandchild.dep_ == "pobj":
                                    objects.append((grandchild, correct_prep))
                            continue
                    
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            objects.append((grandchild, prep_text))
        except (TypeError, AttributeError):
            pass
        return objects

    # ── Improved Verb Triple Extraction ─────────────────────────────────────

    def _verb_triples(self, verb) -> list[Triple]:
        """Extract triples from active verb constructions with improved accuracy."""
        triples = []
        try:
            rel, rel_idx = self._relation_span(verb)
            subjects = self._get_subjects(verb)
            objects = self._get_objects(verb)

            # Relative-clause head as subject: use it when there's no subject at
            # all, AND when the "subject" spaCy found is just the relative
            # pronoun itself (which/who/that) — the common case that was
            # previously missed.
            relcl = self._relcl_subject(verb)
            if relcl and not subjects:
                subjects = [relcl]
            else:
                subjects = self._resolve_relcl_subjects(verb, subjects)

            # Passive: keep the grammatical (surface) subject as arg1 — it is the
            # patient being talked about — and put the by-agent as arg2. Only the
            # relation string needs to carry the passive marker ("... by").
            passive = self._passive_agent(verb)
            if passive:
                agent_text, agent_idx = self._clean_subject_span(passive)
                for subj in subjects:
                    subj_text, subj_idx = self._subtree_span(
                        subj, exclude_deps={"relcl", "acl", "advcl", "ccomp"}
                    )
                    rel_with_by = f"{rel} by" if "by" not in rel else rel
                    triples.append((subj_text, rel_with_by, agent_text, [subj_idx], agent_idx, rel_idx))
                return triples

            # If no subjects, try to use the verb head
            if not subjects and verb.head.pos_ == "VERB":
                subjects = self._get_subjects(verb.head)

            for subj in subjects:
                s_text, s_idx = self._clean_subject_span(subj)
                
                if len(s_text.split()) < self.config.min_subject_words:
                    continue
                    
                if objects:
                    for obj in objects:
                        # Handle tuple case from prepositional objects
                        if isinstance(obj, tuple):
                            o_text, o_idx = self._subtree_span(obj[0], exclude_deps=CLAUSAL_LEAK_DEPS)
                            prep = obj[1]
                            # Fix relation with preposition
                            fixed_rel = self._fix_relation_with_preposition(rel, verb.text, prep)
                            triples.append((s_text, fixed_rel, o_text, [s_idx], o_idx, rel_idx))
                        else:
                            o_text, o_idx = self._subtree_span(obj, exclude_deps=CLAUSAL_LEAK_DEPS)
                            if len(o_text.split()) >= self.config.min_object_words:
                                triples.append((s_text, rel, o_text, [s_idx], o_idx, rel_idx))
                else:
                    prep_objects = self._get_prepositional_objects(verb)
                    if prep_objects:
                        for prep_obj in prep_objects:
                            if isinstance(prep_obj, tuple):
                                o_text, o_idx = self._subtree_span(prep_obj[0], exclude_deps=CLAUSAL_LEAK_DEPS)
                                prep = prep_obj[1]
                                fixed_rel = self._fix_relation_with_preposition(rel, verb.text, prep)
                                triples.append((s_text, fixed_rel, o_text, [s_idx], o_idx, rel_idx))

            triples.extend(self._object_prep_relations(verb, subjects))
            triples.extend(self._xcomp_relations(verb))
            triples.extend(self._handle_conjoined_predicates(verb))
        except (TypeError, AttributeError, ValueError):
            pass

        return triples

    def _object_prep_relations(self, verb, subjects) -> list[Triple]:
        """Extract prepositional relations with proper preposition handling."""
        triples = []
        
        try:
            if not subjects:
                subjects = self._get_subjects(verb)
            
            for prep in verb.children:
                if prep.dep_ != "prep":
                    continue
                for pobj in prep.children:
                    if pobj.dep_ != "pobj":
                        continue
                    o_text, o_idx = self._subtree_span(pobj, exclude_deps=CLAUSAL_LEAK_DEPS)
                    for subj in subjects:
                        s_text, s_idx = self._clean_subject_span(subj)
                        # Fix preposition
                        correct_prep = self._correct_preposition(verb.text, prep.text)
                        if verb.head.dep_ == 'ROOT' and verb.head.text != verb.text:
                            rel = f"{verb.head.text} to {verb.text} {correct_prep}"
                            triples.append((s_text, rel, o_text, [s_idx], o_idx, [verb.head.i, verb.i, prep.i]))
                        else:
                            rel = f"{verb.text} {correct_prep}"
                            triples.append((s_text, rel, o_text, [s_idx], o_idx, [verb.i, prep.i]))
        except (TypeError, AttributeError, ValueError):
            pass
        
        return triples

    def _handle_conjoined_predicates(self, verb) -> list[Triple]:
        """Enhanced handling of conjoined predicates with subject sharing."""
        triples = []
        try:
            subjects = self._get_subjects(verb)
            
            if not subjects and verb.head.pos_ == "VERB":
                subjects = self._get_subjects(verb.head)
            
            all_verbs = [verb] + self._expand_conj(verb)
            
            for conj in all_verbs:
                if conj.pos_ not in ["VERB", "AUX"]:
                    continue
                
                rel, rel_idx = self._relation_span(conj)
                objects = self._get_objects(conj)
                
                for subj in subjects:
                    s_text, s_idx = self._clean_subject_span(subj)
                    
                    if objects:
                        for obj in objects:
                            if isinstance(obj, tuple):
                                o_text, o_idx = self._subtree_span(obj[0])
                                prep = obj[1]
                                fixed_rel = self._fix_relation_with_preposition(rel, conj.text, prep)
                                triples.append((s_text, fixed_rel, o_text, [s_idx], o_idx, rel_idx))
                            else:
                                o_text, o_idx = self._subtree_span(obj, exclude_deps=CLAUSAL_LEAK_DEPS)
                                triples.append((s_text, rel, o_text, [s_idx], o_idx, rel_idx))
                    else:
                        prep_objects = self._get_prepositional_objects(conj)
                        if prep_objects:
                            for prep_obj in prep_objects:
                                if isinstance(prep_obj, tuple):
                                    o_text, o_idx = self._subtree_span(prep_obj[0])
                                    prep = prep_obj[1]
                                    fixed_rel = self._fix_relation_with_preposition(rel, conj.text, prep)
                                    triples.append((s_text, fixed_rel, o_text, [s_idx], o_idx, rel_idx))
                        else:
                            # For motion verbs, add context
                            if conj.text in ["went", "came", "left", "arrived"]:
                                triples.append((s_text, rel, "(action)", [s_idx], [s_idx], rel_idx))
        except (TypeError, AttributeError, ValueError):
            pass
        
        return triples

    # ─── Enhanced Numerical Extraction ──────────────────────────────────────

    def _extract_numerical_relations(self, sent) -> list[Triple]:
        """Enhanced numerical extraction with full context and proper relations."""
        triples = []
        
        try:
            for token in sent:
                if token.ent_type_ in NUMERICAL_ENTITY_TYPES:
                    head = token.head
                    num_phrase = self._get_full_number_phrase(token)
                    
                    # Find appropriate subject
                    subjects = []
                    current = token
                    try:
                        while current.head != current:
                            current = current.head
                            if current.dep_ in SUBJECT_DEPS or current.pos_ in ["NOUN", "PROPN"]:
                                subjects.extend(self._expand_conj(current))
                                break
                    except (TypeError, AttributeError):
                        pass
                    
                    if not subjects and head.pos_ == "VERB":
                        subjects = self._get_subjects(head)
                    
                    if subjects:
                        for subj in subjects:
                            s_text, s_idx = self._clean_subject_span(subj)
                            rel=head.text
                            '''
                            # Determine appropriate relation based on entity type and context
                            if token.ent_type_ == "PERCENT":
                                if any(w in str(head.text).lower() for w in ["growth", "increase", "decline", "fall", "rise"]):
                                    rel = "grew_to"
                                elif any(w in str(head.text).lower() for w in ["was", "is", "are"]):
                                    rel = "was"
                                else:
                                    rel = "had_percent"
                            elif token.ent_type_ == "DATE":
                                rel = "occurred_in"
                            elif token.ent_type_ == "MONEY":
                                rel = "valued_at"
                            elif token.ent_type_ == "CARDINAL":
                                if any(w in str(head.text).lower() for w in ["population", "people", "residents"]):
                                    rel = "has_population"
                                elif any(w in str(head.text).lower() for w in ["hosted", "has", "have"]):
                                    rel = "has"
                                else:
                                    rel = "has_value"
                            else:
                                rel = f"has_{token.ent_type_.lower()}"
                            '''
                            context = self._get_numerical_context(token)
                            full_object = f"{num_phrase}{context}"
                            triples.append((
                                s_text,
                                rel,
                                full_object,
                                [s_idx],
                                (token.left_edge.i, token.right_edge.i + 1),
                                [head.i]
                            ))
                        continue
                    
                    # Fallback: find a subject in the sentence
                    for token2 in sent:
                        if token2.dep_ in SUBJECT_DEPS:
                            s_text, s_idx = self._clean_subject_span(token2)
                            triples.append((
                                s_text,
                                #"has_value",
                                head.text,
                                num_phrase,
                                [s_idx],
                                (token.left_edge.i, token.right_edge.i + 1),
                                [head.i]
                            ))
                            break
        except (TypeError, AttributeError, ValueError):
            pass
        
        return triples

    # ─── Enhanced Copular Extraction ─────────────────────────────────────────

    def _copular_triples(self, sent) -> list[Triple]:
        """Extract triples from copular sentences with improved accuracy."""
        triples = []
        try:
            for token in sent:
                cop = None
                for c in token.children:
                    if c.dep_ == "cop":
                        cop = c
                        break
                if not cop:
                    continue
                
                subjects = []
                for child in token.children:
                    if child.dep_ in SUBJECT_DEPS:
                        subjects.extend(self._expand_conj(child))
                
                if not subjects:
                    try:
                        chunks = list(sent.noun_chunks)
                        for chunk in chunks:
                            if chunk.root.i < token.i:
                                subjects.append(chunk.root)
                    except (TypeError, AttributeError):
                        pass
                
                for subj in subjects:
                    s_text, s_idx = self._clean_subject_span(subj)
                    o_text, o_idx = self._predicate_span(token)
                    triples.append((s_text, cop.text, o_text, [s_idx], o_idx, [cop.i]))
                
                # Handle prepositional predicates (like "is in the Kantō region")
                for prep in token.children:
                    if prep.dep_ == "prep":
                        for pobj in prep.children:
                            if pobj.dep_ == "pobj":
                                o_text, o_idx = self._subtree_span(pobj, exclude_deps=CLAUSAL_LEAK_DEPS)
                                for subj in subjects:
                                    s_text, s_idx = self._clean_subject_span(subj)
                                    # Fix: use 'is in' instead of 'is_in' for location predicates
                                    if prep.text.lower() in ["in", "on", "at", "from", "to"]:
                                        relation = f"{cop.text} {prep.text}"
                                    else:
                                        relation = f"{cop.text} {prep.text}"
                                    triples.append(
                                        (
                                            s_text,
                                            relation,
                                            o_text,
                                            [s_idx],
                                            o_idx,
                                            [cop.i, prep.i]
                                        )
                                    )
        except (TypeError, AttributeError, ValueError):
            pass
        return triples

    # ─── Improved Fallback ───────────────────────────────────────────────────
    def _fallback(self, sent) -> list[Triple]:
        """Improved fallback with better token selection and relation detection."""
        try:
            content_tokens = [t for t in sent if not t.is_punct and t.pos_ not in ["PUNCT", "CCONJ"]]
        
            if len(content_tokens) < 2:
                return []
        
        # Find the main verb with priority for action verbs
            verb_candidates = [
                t for t in content_tokens 
                if t.tag_.startswith('VB') or t.text.lower() in ['will', 'would', 'be', 'am', 'are', 'is', 'was', 'were', 'have', 'has', 'had']
            ]
        
        # If no verb, create a simple triple
            if not verb_candidates:
                if len(content_tokens) >= 2:
                    subject = " ".join(t.text for t in content_tokens[:-1])
                    predicate = content_tokens[-1].text
                    s_idx = [t.i for t in content_tokens[:-1]]
                    o_idx = [content_tokens[-1].i]
                    rel_idx = [content_tokens[-1].i]
                    return [(subject, "is", predicate, s_idx, o_idx, rel_idx)]
                return []
        
        # Prefer main verb (last verb in sentence)
            verb_candidates.sort(key=lambda t: t.i)
            main_verb = verb_candidates[-1]
        
        # Get subject (noun before verb)
            subject_tokens = [t for t in content_tokens if t.i < main_verb.i and t.dep_ not in ["aux", "auxpass"]]
            object_tokens = [t for t in content_tokens if t.i > main_verb.i]
        
        # Try noun chunks if no subject found
            if not subject_tokens:
                try:
                    chunks = list(sent.noun_chunks)
                    for chunk in chunks:
                        if chunk.start < main_verb.i:
                            subject_tokens = [t for t in chunk]
                            break
                except (TypeError, AttributeError):
                    pass
        
            triples = []
        
            if subject_tokens and object_tokens:
                subject = " ".join(t.text for t in subject_tokens if not t.is_punct)
            
            # Build relation with prepositions
                relation = main_verb.text
                prep_tokens = []
                prep_indices = []
            
            # Check for prepositions and their objects
                for child in main_verb.children:
                    if child.dep_ == "prep":
                        prep_text = child.text
                    # Get the object of the preposition
                        prep_obj = None
                        for grandchild in child.children:
                            if grandchild.dep_ in ["pobj", "dobj"]:
                                prep_obj = grandchild.text
                            # Add preposition object to object tokens if not already there
                                if grandchild not in object_tokens:
                                    object_tokens.append(grandchild)
                                break
                    
                    # Get the corrected preposition
                        correct_prep = self._correct_preposition(main_verb.text, prep_text)
                        prep_tokens.append(correct_prep)
                        prep_indices.append(child.i)
                    
                    # If preposition has an object, include it in relation
                        if prep_obj:
                            relation = f"{relation} {correct_prep}"
                        else:
                            relation = f"{relation} {correct_prep}"
            
            # Build object with proper filtering
                obj_tokens = [t for t in object_tokens if not t.is_punct and t.pos_ not in ["PUNCT", "ADP"]]
                if not obj_tokens and prep_tokens:
                # If only prepositions, use them as object
                    obj_tokens = [t for t in object_tokens if t.pos_ == "ADP"]
            
                obj = " ".join(t.text for t in obj_tokens) if obj_tokens else ""
            
            # Get indices
                s_idx = [t.i for t in subject_tokens if not t.is_punct]
                o_idx = [t.i for t in obj_tokens] if obj_tokens else []
                rel_idx = [main_verb.i] + prep_indices if prep_indices else [main_verb.i]
            
                if subject and relation and obj:
                    triples.append((subject, relation, obj, s_idx, o_idx, rel_idx))
                
            elif subject_tokens:
                subject = " ".join(t.text for t in subject_tokens if not t.is_punct)
                s_idx = [t.i for t in subject_tokens if not t.is_punct]
                rel_idx = [main_verb.i]
            
                if subject and main_verb.text:
                # Try to find a meaningful object from context
                    object_text = None
                    o_idx = []
                
                # Look for noun chunks after verb
                    try:
                        for chunk in sent.noun_chunks:
                            if chunk.start > main_verb.i:
                                object_text = " ".join(t.text for t in chunk)
                                o_idx = [t.i for t in chunk]
                                break
                    except (TypeError, AttributeError):
                        pass
                
                # Look for any content token after verb
                    if not object_text:
                        after_verb = [t for t in content_tokens if t.i > main_verb.i]
                        if after_verb:
                        # Find the first noun or adjective after verb
                            for t in after_verb:
                                if t.pos_ in ["NOUN", "PROPN", "ADJ"]:
                                    object_text = t.text
                                    o_idx = [t.i]
                                    break
                        # If no noun/adjective, use first token
                            if not object_text:
                                object_text = after_verb[0].text
                                o_idx = [after_verb[0].i]
                
                # If still no object, look for object in relation with preposition
                    if not object_text:
                        for child in main_verb.children:
                            if child.dep_ == "prep":
                                for grandchild in child.children:
                                    if grandchild.dep_ in ["pobj", "dobj"]:
                                        object_text = grandchild.text
                                        o_idx = [grandchild.i]
                                        break
                                if object_text:
                                    break
                
                # Use "something" as last resort
                    if not object_text:
                        object_text = "something"
                
                    triples.append((subject, main_verb.text, object_text, s_idx, o_idx, rel_idx))
        
            return triples
        
        except (TypeError, AttributeError, ValueError):
            return []    

    
    # ─── Enhanced Output Cleaning ────────────────────────────────────────────
    def _clean_output_triples(self, triples: list[Triple]) -> list[Triple]:
        """Clean up output triples with better filtering."""
        cleaned = []
    
        for t in triples:
            try:
                subject, relation, obj, *rest = t
            
             # تحويل جميع القيم إلى نص
                subject = str(subject) if subject is not None else ""
                relation = str(relation) if relation is not None else ""
                obj = str(obj) if obj is not None else ""
            
                # تصفية الكائنات غير المرغوب فيها
                forbidden_patterns = [
                    "<_cython_3_2_4.generator",
                    "<generator object",
                    "generator>",
                    "<cython",
                    "object at 0x",
                    "built-in method"
                ]
            
                # فحص كل حقل للتأكد من عدم احتوائه على أنماط محظورة
                triple_text = f"{subject}{relation}{obj}"
                if any(pattern in triple_text for pattern in forbidden_patterns):
                    continue
            
                # Clean text
                subject = ' '.join(subject.split())
                relation = ' '.join(relation.split())
                obj = ' '.join(obj.split())
            
                # Remove empty or trivial triples
                if not subject or not obj or subject == obj:
                    continue
            
                if obj.lower() == "true" and relation.lower() == "exists":
                    continue
            
                if subject.lower() == obj.lower():
                    continue
            
                if len(obj) < 2:
                    continue
            
                # Fix common relation issues
                relation = relation.replace('  ', ' ')
                relation = relation.replace(' to to ', ' to ')
                relation = relation.replace(' of of ', ' of ')
                relation = relation.replace(' in in ', ' in ')
                relation = relation.replace(' on on ', ' on ')
                relation = relation.replace(' from from ', ' from ')
            
                # Remove trailing prepositions that don't make sense
                trailing_preps = ['with', 'to', 'for', 'of', 'about']
                for prep in trailing_preps:
                    if relation.endswith(f' {prep}') and not relation.startswith(prep):
                        if not any(word in obj.lower() for word in [prep]):
                            relation = relation.rsplit(' ', 1)[0]
            
                if len(relation) < 2:
                    continue
            
                cleaned.append((subject, relation, obj, *rest))
            except (TypeError, AttributeError, IndexError) as e:
                # تخطي الثلاثيات التي تسبب أخطاء
                continue
    
        return cleaned
    
    # ─── Enhanced Deduplication ─────────────────────────────────────────────

    def _deduplicate(self, triples: list[Triple]) -> list[Triple]:
        """Enhanced deduplication with better semantic similarity."""
        if not triples:
            return []
        
        seen = set()
        stage1: list[Triple] = []
        for t in triples:
            try:
                key = (t[0].lower().strip(), t[1].lower().strip(), t[2].lower().strip())
                if key in seen or not all(t[0:3]):
                    continue
                seen.add(key)
                stage1.append(t)
            except (TypeError, AttributeError):
                continue
        
        if self.config.deduplication_level == 'strict':
            return stage1
        
        # Aggressive deduplication: group by normalized subject and object
        normalized_groups: Dict[tuple, list[Triple]] = {}
        order: list = []
        
        for t in stage1:
            try:
                subj_words = t[0].lower().split()
                obj_words = t[2].lower().split()
                
                # Remove stopwords for better matching
                stopwords = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'it', 'he', 'she', 'they', 'we', 'you', 'its'}
                subj_norm = ' '.join([w for w in subj_words if w not in stopwords])
                obj_norm = ' '.join([w for w in obj_words if w not in stopwords])
                
                if not subj_norm:
                    subj_norm = ' '.join(subj_words)
                if not obj_norm:
                    obj_norm = ' '.join(obj_words)
                
                key = (subj_norm, obj_norm)
                if key not in normalized_groups:
                    order.append(key)
                    normalized_groups[key] = []
                normalized_groups[key].append(t)
            except (TypeError, AttributeError):
                continue
        
        results: list[Triple] = []
        for key in order:
            group = normalized_groups.get(key, [])
            if len(group) == 1:
                results.append(group[0])
                continue
            
            # Keep the most informative triple
            try:
                group.sort(key=lambda x: (len(x[1].split()), len(x[0]) + len(x[2])), reverse=True)
            except (TypeError, AttributeError):
                results.extend(group)
                continue
            
            kept = []
            for t in group:
                try:
                    rel_tokens = set(t[1].lower().split())
                    is_subsumed = False
                    for kept_t in kept:
                        kept_rel_tokens = set(kept_t[1].lower().split())
                        if rel_tokens.issubset(kept_rel_tokens) and len(rel_tokens) < len(kept_rel_tokens):
                            is_subsumed = True
                            break
                    if not is_subsumed:
                        kept.append(t)
                except (TypeError, AttributeError):
                    kept.append(t)
            
            results.extend(kept)
        
        return results

    # ─── Quality Score Calculation ──────────────────────────────────────────

    def _calculate_quality_score(self, triple: Triple, doc) -> float:
        """Calculate quality score for a triple with enhanced criteria."""
        try:
            subject, relation, obj = triple[0], triple[1], triple[2]
        except (TypeError, IndexError):
            return 0.0
        
        score = 0.0
        
        # Subject quality
        if len(subject.split()) >= 2:
            score += 0.2
        elif len(subject.split()) >= 1:
            score += 0.1
        
        # Relation quality
        rel_tokens = relation.split()
        if len(rel_tokens) >= 2:
            score += 0.2
        elif len(rel_tokens) >= 1:
            score += 0.1
        
        # Check if relation is a proper predicate (not too short)
        if relation and relation != 'is' and len(relation) > 2:
            score += 0.1
        
        # Object quality
        if len(obj.split()) >= 3:
            score += 0.3
        elif len(obj.split()) >= 2:
            score += 0.2
        elif len(obj.split()) >= 1:
            score += 0.1
        
        # Index quality
        if triple[3] and triple[4]:
            score += 0.2
        
        return min(score, 1.0)

    # ─── Main Extraction ─────────────────────────────────────────────────────

    def extract(self, doc) -> tuple[list[Triple], list]:
        """
        Extract (subject, relation, object) triples from *doc*.

        Returns a deduplicated list of 6-tuples:
          (subj_text, rel_text, obj_text, subj_idx, obj_idx, rel_idx)
        """
        self._span_cache.clear()
        self._relation_cache.clear()
        self._subject_cache.clear()
        self._noun_chunk_cache.clear()
        self._num_phrase_cache.clear()

        try:
            resolved = self._resolve_coref_improved(doc)
            if resolved != doc.text:
                doc = self.nlp(resolved)
        except (TypeError, AttributeError, ValueError):
            pass

        triples: list[Triple] = []
        token_lists = []

        for sent in doc.sents:
            sent_triples = self._process_sentence_with_gap_fill(sent)
            
            if len(sent_triples) > self.config.max_triples_per_sentence:
                sent_triples = sent_triples[:self.config.max_triples_per_sentence]
            
            triples.extend(sent_triples)
            token_lists.append([token.text for token in sent])

        if self.config.extract_cross_sentence:
            triples.extend(self._extract_cross_sentence_relations(doc))

        triples = self._deduplicate(triples)
        
        if self.config.clean_output:
            triples = self._clean_output_triples(triples)

        return triples, token_lists

    # ─── Helper Methods ──────────────────────────────────────────────────────

    def _subtree_span(self, token, exclude_deps: set[str] | None = None, max_len: int = 50) -> tuple[str, tuple[int, int]]:
        """Return the text and (start, end) of *token*'s subtree."""
        cache_key = (token.i, id(token.doc), tuple(sorted(exclude_deps or [])))
        if cache_key in self._span_cache:
            return self._span_cache[cache_key]
        
        if exclude_deps is None:
            exclude_deps = set()

        # Named entity → keep the full entity span
        if token.ent_type_:
            s = token.left_edge.i
            e = token.right_edge.i + 1
            # Expand for multi-word entities
            for i in range(e, min(e + 3, len(token.doc))):
                if token.doc[i].ent_type_ == token.ent_type_:
                    e = i + 1
                else:
                    break
            result = _clean(token.doc[s:e].text), (s, e)
            self._span_cache[cache_key] = result
            return result

        # Get subtree nodes
        nodes: list = []
        try:
            subtree_list = list(token.subtree)
            for t in subtree_list:
                if t.dep_ in exclude_deps:
                    continue
                if t.is_punct:
                    continue
                nodes.append(t)
        except (TypeError, AttributeError):
            nodes = [token]

        if not nodes:
            result = token.text, (token.i, token.i + 1)
            self._span_cache[cache_key] = result
            return result

        nodes.sort(key=lambda x: x.i)
        if len(nodes) > max_len:
            nodes = nodes[:max_len]

        s, e = nodes[0].i, nodes[-1].i + 1
        result = _clean(token.doc[s:e].text), (s, e)
        self._span_cache[cache_key] = result
        return result

    def _get_noun_chunk(self, token):
        """Get the noun chunk containing the token."""
        doc = token.doc
        
        if token.i in self._noun_chunk_cache:
            return self._noun_chunk_cache[token.i]
        
        try:
            chunks = list(doc.noun_chunks)
            for chunk in chunks:
                if chunk.start <= token.i < chunk.end:
                    self._noun_chunk_cache[token.i] = chunk
                    return chunk
        except (TypeError, AttributeError):
            pass
        
        self._noun_chunk_cache[token.i] = None
        return None

    def _get_subjects(self, verb) -> list:
        """Get all subjects for a verb."""
        subjects = []
        seen = set()
        
        try:
            children_list = list(verb.children)
            for child in children_list:
                if child.dep_ in SUBJECT_DEPS and child.i not in seen:
                    seen.add(child.i)
                    subjects.extend(self._expand_conj(child))
            
            if not subjects:
                relcl = self._relcl_subject(verb)
                if relcl and relcl.i not in seen:
                    seen.add(relcl.i)
                    subjects.append(relcl)
            else:
                # Found subjects — but if they're literally the relative pronoun
                # (which/who/that) heading this relcl/acl verb, swap in the noun
                # it actually refers to, so downstream triples don't read
                # ('which', ...) / ('that', ...).
                subjects = self._resolve_relcl_subjects(verb, subjects)

            if not subjects:
                for ancestor in verb.ancestors:
                    if ancestor.pos_ == "VERB":
                        for child in ancestor.children:
                            if child.dep_ in SUBJECT_DEPS and child.i not in seen:
                                seen.add(child.i)
                                subjects.extend(self._expand_conj(child))
                        if subjects:
                            break
            
            if not subjects:
                try:
                    chunks = list(verb.doc.noun_chunks)
                    for chunk in chunks:
                        if chunk.root.i < verb.i and chunk.root.i not in seen:
                            for token in chunk:
                                if token.dep_ in ["nsubj", "nsubjpass"]:
                                    seen.add(token.i)
                                    subjects.append(token)
                                    break
                except (TypeError, AttributeError):
                    pass
        except (TypeError, AttributeError):
            pass
        
        return subjects

    def _expand_conj(self, token) -> list:
        """BFS over conjuncts."""
        results, visited, queue = [], set(), deque([token])
        while queue:
            t = queue.popleft()
            if t.i in visited:
                continue
            visited.add(t.i)
            results.append(t)
            try:
                for c in t.conjuncts:
                    if c.i not in visited:
                        queue.append(c)
            except (TypeError, AttributeError):
                pass
        return results

    RELATIVE_PRONOUNS = {"who", "whom", "whose", "which", "that"}

    def _resolve_relcl_subjects(self, verb, subjects):
        """If `subjects` is just the relative pronoun heading this relcl/acl verb
        (e.g. 'which', 'who', 'that'), replace it with the antecedent noun phrase
        the clause actually modifies, so the extracted triple is interpretable
        outside the clause instead of reading e.g. ('which', 'came to rest on', ...).
        """
        if not subjects:
            return subjects
        if not all(
            t.dep_ in SUBJECT_DEPS and t.text.lower() in self.RELATIVE_PRONOUNS
            for t in subjects
        ):
            return subjects
        head_noun = self._relcl_subject(verb)
        return [head_noun] if head_noun else subjects

    def _relcl_subject(self, verb):
        """Find the noun a relative/adjectival clause modifies."""
        try:
            if verb.dep_ in {"relcl", "acl"}:
                head = verb.head
                seen = {verb.i}
                while head.dep_ in {"relcl", "acl"} and head.i not in seen:
                    seen.add(head.i)
                    head = head.head
                return head
            for child in verb.children:
                if child.dep_ in {"relcl", "acl"}:
                    return child.head
        except (TypeError, AttributeError):
            pass
        return None

    def _passive_agent(self, verb):
        """Find the agent in a passive construction."""
        try:
            for child in verb.children:
                if child.dep_ == "agent":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            return pobj
        except (TypeError, AttributeError):
            pass
        return None

    def _predicate_span(self, token) -> tuple[str, tuple[int, int]]:
        """Return the predicate span for copular sentences."""
        nodes = []
        try:
            subtree_list = list(token.subtree)
            for t in subtree_list:
                if t.dep_ not in SUBJECT_DEPS and t.dep_ != "cop" and not t.is_punct:
                    nodes.append(t)
        except (TypeError, AttributeError):
            nodes = [token]
            
        if not nodes:
            return token.text, (token.i, token.i + 1)

        nodes.sort(key=lambda x: x.i)
        s, e = nodes[0].i, nodes[-1].i + 1
        return _clean(token.doc[s:e].text), (s, e)

    def _get_full_number_phrase(self, token) -> str:
        """Get complete number phrase including modifiers and units."""
        cache_key = token.i
        if cache_key in self._num_phrase_cache:
            return self._num_phrase_cache[cache_key]
        
        doc = token.doc
        
        left = token.left_edge.i
        right = token.right_edge.i + 1
        
        for i in range(right, min(right + 4, len(doc))):
            try:
                text_lower = doc[i].text.lower()
                if doc[i].is_alpha and text_lower in ["percent", "million", "billion", "trillion", "thousand", "hundred"]:
                    right = i + 1
                elif doc[i].is_punct and doc[i].text in [",", "."]:
                    if i + 1 < len(doc) and doc[i + 1].text.lower() in ["percent", "million", "billion"]:
                        right = i + 2
                        break
                    continue
                else:
                    break
            except (TypeError, AttributeError):
                break
        
        for i in range(left - 1, max(left - 3, -1), -1):
            try:
                if doc[i].text in ["$", "€", "£", "¥", "¢"]:
                    left = i
                    break
            except (TypeError, AttributeError):
                break
        
        result = _clean(doc[left:right].text)
        self._num_phrase_cache[cache_key] = result
        return result

    def _get_numerical_context(self, token) -> str:
        """Get context for numerical values."""
        doc = token.doc
        context_parts = []
        
        try:
            for i in range(token.i - 2, token.i + 3):
                if i < 0 or i >= len(doc) or i == token.i:
                    continue
                if doc[i].text.lower() in ["of", "for", "by", "from", "to"]:
                    context_parts.append(doc[i].text)
                    if i + 1 < len(doc) and doc[i + 1].pos_ in ["NOUN", "PROPN"]:
                        context_parts.append(doc[i + 1].text)
                        break
        except (TypeError, AttributeError):
            pass
        
        return " " + " ".join(context_parts) if context_parts else ""

    def _xcomp_relations(self, verb) -> list[Triple]:
        """Extract triples from xcomp/ccomp constructions."""
        triples = []
        try:
            subjects = self._get_subjects(verb)
            for child in verb.children:
                if child.dep_ not in CLAUSAL_COMPLEMENT:
                    continue
                rel, rel_idx = self._relation_span(child)
                objects = self._get_objects(child)
                if not objects:
                    objects2 = self._get_objects_alt(child)
                for subj in subjects:
                    s_text, s_idx = self._clean_subject_span(subj)
                    if objects:
                        for obj in objects:
                            o_text, o_idx = self._subtree_span(obj, exclude_deps=CLAUSAL_LEAK_DEPS)
                            triples.append((s_text, rel, o_text, [s_idx], o_idx, rel_idx))
                    elif objects2:
                        rel2 = f"{verb.text} to {child.text}"
                        for obj in objects2:
                            o_text, o_idx = self._subtree_span(obj, exclude_deps=CLAUSAL_LEAK_DEPS)
                            triples.append((s_text, rel2, o_text, [s_idx], o_idx, [verb.i, child.i]))
                    else:
                        for prep in child.children:
                            if prep.dep_ == "prep":
                                for pobj in prep.children:
                                    if pobj.dep_ == "pobj":
                                        o_text, o_idx = self._subtree_span(pobj, exclude_deps=CLAUSAL_LEAK_DEPS)
                                        triples.append(
                                            (
                                                s_text,
                                                f"{verb.text} to {child.text} {prep.text}",
                                                o_text,
                                                [s_idx],
                                                o_idx,
                                                [verb.i, child.i, prep.i]
                                            )
                                        )
        except (TypeError, AttributeError, ValueError):
            pass
        return triples

    def _get_objects_alt(self, verb) -> list:
        """Alternative object getter for xcomp/ccomp."""
        objects = []
        try:
            for child in verb.children:
                if child.dep_ in {"dobj", "obj"}:
                    objects.extend(self._expand_conj(child))
        except (TypeError, AttributeError):
            pass
        return objects

    def _apposition_triples(self, sent) -> list[Triple]:
        """Extract triples from appositions."""
        triples = []
        try:
            for token in sent:
                if token.dep_ != "appos":
                    continue
                s_text, s_idx = self._clean_subject_span(token.head)
                o_text, o_idx = self._subtree_span(token)
                triples.append((s_text, "is", o_text, [s_idx], o_idx, [token.i]))
        except (TypeError, AttributeError, ValueError):
            pass
        return triples

    def _extract_complex_appositions(self, sent) -> list[Triple]:
        """Extract complex appositions including 'X, known as Y' patterns."""
        triples = []
        
        try:
            for token in sent:
                if token.dep_ == "acl" and token.text in ["known", "called", "named", "also known"]:
                    head_text, head_idx = self._clean_subject_span(token.head)
                    rel = "is_known_as"
                    for child in token.children:
                        if child.dep_ in ["ccomp", "xcomp"]:
                            o_text, o_idx = self._subtree_span(child)
                            triples.append((head_text, rel, o_text, [head_idx], o_idx, [token.i, child.i]))
                
                if token.dep_ == "relcl" and token.pos_ == "VERB":
                    head_text, head_idx = self._clean_subject_span(token.head)
                    rel, rel_idx = self._relation_span(token)
                    objects = self._get_objects(token)
                    
                    for obj in objects:
                        o_text, o_idx = self._subtree_span(obj, exclude_deps=CLAUSAL_LEAK_DEPS)
                        triples.append((head_text, rel, o_text, [head_idx], o_idx, rel_idx))
        except (TypeError, AttributeError, ValueError):
            pass
        
        return triples

    def _acl_triples(self, token) -> list[Triple]:
        """Extract triples from adjectival clauses."""
        triples = []
        try:
            if token.dep_ != "acl":
                return triples

            s_text, s_idx = self._clean_subject_span(token.head)
            rel, rel_idx = self._relation_span(token)

            for child in token.children:
                if child.dep_ in {"obj", "dobj", "attr"}:
                    o_text, o_idx = self._subtree_span(child)
                    triples.append((s_text, rel, o_text, [s_idx], o_idx, rel_idx))
                elif child.dep_ == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            o_text, o_idx = self._subtree_span(pobj, exclude_deps=CLAUSAL_LEAK_DEPS)
                            prep_rel = f"{rel} {child.text}"
                            triples.append(
                                (s_text, prep_rel, o_text,
                                 [s_idx], o_idx, rel_idx + [child.i])
                            )
        except (TypeError, AttributeError, ValueError):
            pass
        return triples

    def _advcl_triples(self, verb) -> list[Triple]:
        """Extract triples from adverbial clauses."""
        triples = []
        try:
            subjects = self._get_subjects(verb)
            for child in verb.children:
                if child.dep_ != "advcl":
                    continue
                rel, rel_idx = self._relation_span(child)
                mark = None
                for c in child.children:
                    if c.dep_ == "mark":
                        mark = c.text
                        break
                combined_rel = f"{mark} {rel}".strip() if mark else rel
                for subj in subjects:
                    s_text, s_idx = self._clean_subject_span(subj)
                    o_text, o_idx = self._subtree_span(child)
                    triples.append((s_text, combined_rel, o_text, [s_idx], o_idx, rel_idx))
        except (TypeError, AttributeError, ValueError):
            pass
        return triples

    def _possessive_triples(self, sent) -> list[Triple]:
        """Extract possessive relations."""
        triples = []
        try:
            for token in sent:
                if token.dep_ == "poss" or token.dep_ == "possessive":
                    owner_text, owner_idx = self._clean_subject_span(token)
                    possessed_text, possessed_idx = self._clean_subject_span(token.head)
                    triples.append(
                        (owner_text, "has", possessed_text,
                         [owner_idx], possessed_idx, [token.i])
                    )
        except (TypeError, AttributeError, ValueError):
            pass
        return triples

    def _nmod_compound_triples(self, sent) -> list[Triple]:
        """Extract relations from noun modifiers and compounds."""
        triples = []
        try:
            for token in sent:
                if token.dep_ in {"nmod", "nmod:poss"}:
                    for prep in token.children:
                        if prep.dep_ == "prep" and prep.text.lower() == "of":
                            for pobj in prep.children:
                                if pobj.dep_ == "pobj":
                                    o_text, o_idx = self._subtree_span(pobj, exclude_deps=CLAUSAL_LEAK_DEPS)
                                    s_text, s_idx = self._clean_subject_span(token)
                                    triples.append(
                                        (o_text, f"has_{token}", s_text,
                                         [o_idx], s_idx, [prep.i])
                                    )
        except (TypeError, AttributeError, ValueError):
            pass
        return triples

    def _attributive_adj_triples(self, sent) -> list[Triple]:
        """Extract triples from attributive adjectives."""
        triples = []
        try:
            for token in sent:
                if token.pos_ == "ADJ" and token.dep_ == "amod":
                    s_text, s_idx = self._clean_subject_span(token.head)
                    triples.append(
                        (s_text, "is", token.text,
                         [s_idx], (token.i, token.i + 1), [token.i])
                    )
        except (TypeError, AttributeError, ValueError):
            pass
        return triples

    def _existential_triples(self, sent) -> list[Triple]:
        """Extract triples from existential constructions."""
        triples = []
        try:
            for token in sent:
                if token.dep_ == "expl" and token.text.lower() == "there":
                    verb = token.head
                    for child in verb.children:
                        if child.dep_ in {"attr", "nsubj"}:
                            o_text, o_idx = self._subtree_span(child)
                            triples.append(
                                (o_text, "exists", "true",
                                 [o_idx], (token.i, token.i + 1), [verb.i])
                            )
        except (TypeError, AttributeError, ValueError):
            pass
        return triples

    def _extract_reported_speech(self, sent) -> list[Triple]:
        """Extract triples from quoted/reported speech."""
        triples = []
        
        try:
            for token in sent:
                if token.dep_ == "ccomp" and token.head.pos_ == "VERB":
                    if token.head.text.lower() in REPORTING_VERBS:
                        subjects = self._get_subjects(token.head)
                        
                        for subj in subjects:
                            s_text, s_idx = self._clean_subject_span(subj)
                            rel = f"{token.head.text}_that"
                            o_text, o_idx = self._subtree_span(token)
                            triples.append((s_text, rel, o_text, [s_idx], o_idx, [token.head.i, token.i]))
                
                if token.text in ['"', "'", "“", "”", "‘", "’"] and token.i + 1 < len(sent):
                    quote_end = None
                    matching_quote = {'"': '"', "'": "'", "“": "”", "”": "“", "‘": "’", "’": "‘"}.get(token.text, token.text)
                    
                    for i in range(token.i + 1, len(sent)):
                        if sent[i].text == matching_quote:
                            quote_end = i
                            break
                    
                    if quote_end and quote_end > token.i + 1:
                        quote_text = _clean(sent[token.i + 1:quote_end])
                        if len(quote_text) > 3:
                            speaker = None
                            for head in token.ancestors:
                                if head.pos_ == "VERB":
                                    for child in head.children:
                                        if child.dep_ in SUBJECT_DEPS:
                                            speaker = child
                                            break
                                    break
                            
                            if speaker:
                                s_text, s_idx = self._clean_subject_span(speaker)
                                triples.append((s_text, "said", quote_text, [s_idx], (token.i, quote_end + 1), [head.i]))
        except (TypeError, AttributeError, ValueError):
            pass
        
        return triples

    def _extract_noun_phrase_relations(self, sent) -> list[Triple]:
        """Extract relations from complex noun phrases."""
        triples = []
        try:
            for token in sent:
                if token.pos_ == "NOUN" and token.dep_ in ["nsubj", "dobj", "pobj"]:
                    np_tokens = [t for t in token.subtree if t.dep_ not in FORBIDDEN_SUBJECT_DEPS]
                    if len(np_tokens) > 1:
                        np_text = " ".join([t.text for t in np_tokens if not t.is_punct])
                        np_start = np_tokens[0].i if np_tokens else token.i
                        np_end = np_tokens[-1].i + 1 if np_tokens else token.i + 1

                        if token.head.pos_ == "VERB":
                            rel, _ = self._relation_span(token.head)
                            for child in token.head.children:
                                if child != token and child.pos_ == "NOUN":
                                    child_np = " ".join([t.text for t in child.subtree if not t.is_punct])
                                    if token.dep_ == "nsubj":
                                        triples.append((
                                            np_text, 
                                            rel, 
                                            child_np,
                                            (np_start, np_end),
                                            (child.i, child.i + 1),
                                            [token.head.i]
                                        ))
        except (TypeError, AttributeError, ValueError):
            pass
        return triples

    def _handle_complex_noun_phrases(self, sent) -> list[Triple]:
        """Extract relations from complex noun phrases."""
        triples = []
        
        try:
            for token in sent:
                if token.dep_ == "appos":
                    head_text, head_idx = self._clean_subject_span(token.head)
                    appos_text, appos_idx = self._subtree_span(token)
                    triples.append((head_text, "is", appos_text, [head_idx], appos_idx, [token.i]))
                
                if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                    for child in token.head.children:
                        if child.dep_ == "conj":
                            conj_subjects = self._get_subjects(child)
                            for subj in conj_subjects:
                                s_text, s_idx = self._clean_subject_span(subj)
                                rel, rel_idx = self._relation_span(child)
                                objects = self._get_objects(child)
                                for obj in objects:
                                    o_text, o_idx = self._subtree_span(obj, exclude_deps=CLAUSAL_LEAK_DEPS)
                                    triples.append((s_text, rel, o_text, [s_idx], o_idx, rel_idx))
        except (TypeError, AttributeError, ValueError):
            pass
        
        return triples

    def _extract_relative_clauses(self, sent) -> list[Triple]:
        """Enhanced extraction of relative clauses."""
        triples = []
        
        try:
            for token in sent:
                if token.dep_ in ["relcl", "acl"] and token.pos_ == "VERB":
                    head_noun = self._relcl_subject(token)
                    if not head_noun:
                        head_noun = token.head
                    
                    rel, rel_idx = self._relation_span(token)
                    objects = self._get_objects(token)
                    
                    subj_tokens = self._get_subjects(token)
                    if not subj_tokens and head_noun:
                        subj_tokens = [head_noun]
                    
                    for subj in subj_tokens:
                        s_text, s_idx = self._clean_subject_span(subj)
                        
                        if objects:
                            for obj in objects:
                                o_text, o_idx = self._subtree_span(obj, exclude_deps=CLAUSAL_LEAK_DEPS)
                                triples.append((s_text, rel, o_text, [s_idx], o_idx, rel_idx))
                        else:
                            for child in token.children:
                                if child.dep_ == "xcomp":
                                    o_text, o_idx = self._subtree_span(child)
                                    triples.append((s_text, f"{rel} to", o_text, [s_idx], o_idx, rel_idx))
                    
                    if head_noun:
                        s_text, s_idx = self._clean_subject_span(head_noun)
                        rel_text = "has_relative_clause"
                        o_text = _clean(token.subtree)
                        triples.append((s_text, rel_text, o_text, [s_idx], (token.left_edge.i, token.right_edge.i + 1), rel_idx))
        except (TypeError, AttributeError, ValueError):
            pass
        
        return triples

    def _extract_cross_sentence_relations(self, doc) -> list[Triple]:
        """Extract relations that span multiple sentences."""
        triples = []
        
        try:
            subjects_map = {}
            for sent in doc.sents:
                for token in sent:
                    if token.dep_ in SUBJECT_DEPS and token.pos_ in ["NOUN", "PROPN"]:
                        key = token.text.lower()
                        if key not in subjects_map:
                            subjects_map[key] = []
                        subjects_map[key].append(token)
            
            for sent_idx, sent in enumerate(doc.sents):
                for token in sent:
                    if token.text.lower() in ["he", "she", "it", "they", "we", "you", "he's", "she's"]:
                        for prev_sent in doc.sents[:sent_idx]:
                            for prev_token in prev_sent:
                                if prev_token.pos_ in ["NOUN", "PROPN"]:
                                    s_text, s_idx = self._clean_subject_span(prev_token)
                                    if token.head.pos_ == "VERB":
                                        rel, rel_idx = self._relation_span(token.head)
                                        objects = self._get_objects(token.head)
                                        for obj in objects:
                                            o_text, o_idx = self._subtree_span(obj, exclude_deps=CLAUSAL_LEAK_DEPS)
                                            triples.append((s_text, rel, o_text, [s_idx], o_idx, rel_idx))
                                        break
        except (TypeError, AttributeError, ValueError):
            pass
        
        return triples

    def _extract_with_context(self, sent) -> list[Triple]:
        """Extract triples while preserving context."""
        triples = []
        
        try:
            main_clause = None
            for token in sent:
                if token.dep_ == "ROOT":
                    main_clause = token
                    break
            
            if not main_clause:
                return triples
            
            rel, rel_idx = self._relation_span(main_clause)
            subjects = self._get_subjects(main_clause)
            objects = self._get_objects(main_clause)
            
            for subj in subjects:
                s_text, s_idx = self._clean_subject_span(subj)
                context = self._get_subordinate_context(main_clause)
                
                for obj in objects:
                    o_text, o_idx = self._subtree_span(obj, exclude_deps=CLAUSAL_LEAK_DEPS)
                    combined_text = f"{s_text} {rel} {o_text}{context}"
                    triples.append((s_text, rel, combined_text, [s_idx], o_idx, rel_idx))
        except (TypeError, AttributeError, ValueError):
            pass
        
        return triples

    def _get_subordinate_context(self, verb) -> str:
        """Get context from subordinate clauses."""
        context_parts = []
        
        try:
            for child in verb.children:
                if child.dep_ in ["advcl", "relcl", "acl"]:
                    clause_text = _clean(child.subtree)
                    if clause_text:
                        context_parts.append(f" ({clause_text})")
        except (TypeError, AttributeError):
            pass
        
        return " ".join(context_parts)

    def _extract_triple_and_handler(self, doc) -> list[Triple]:
        """Simple extractor for basic patterns (used as gap-filler)."""
        triplets = []

        try:
            for token in doc:
                if token.dep_ in ["nsubj", "nsubjpass"] and token.head.pos_ == "VERB":
                    subject_tokens = self._extract_phrase_tokens(token)
                    subject_text = " ".join([t.text for t in subject_tokens])
                    subject_idx = [(subject_tokens[0].i, subject_tokens[-1].i + 1)]

                    verb_token = token.head
                    verb_text = verb_token.text

                    obj_tokens = None
                    relation_text = None
                    obj_idx = None

                    for child in verb_token.children:
                        if child.dep_ in ["dobj", "attr"]:
                            obj_tokens = self._extract_phrase_tokens(child)
                            relation_text = verb_text
                            obj_idx = (child.i, child.i + 1)
                            break

                        elif child.dep_ == "prep":
                            prep = child.text
                            for prep_child in child.children:
                                if prep_child.dep_ == "pobj":
                                    obj_tokens = self._extract_phrase_tokens(prep_child)
                                    relation_text = f"{verb_text} {prep}"
                                    obj_idx = (prep_child.i, prep_child.i + 1)
                                    break
                            if obj_tokens:
                                break

                    if token.dep_ == "nsubjpass":
                        for child in verb_token.children:
                            if child.dep_ == "agent" and child.text == "by":
                                for agent_child in child.children:
                                    if agent_child.dep_ == "pobj":
                                        obj_tokens = self._extract_phrase_tokens(agent_child)
                                        relation_text = f"is {verb_text} by"
                                        obj_idx = (agent_child.i, agent_child.i + 1)
                                        break

                    if obj_tokens and subject_tokens and relation_text:
                        obj_text = " ".join([t.text for t in obj_tokens])
                        rel_idx = [verb_token.i]
                        triplets.append((subject_text, relation_text, obj_text,
                                       subject_idx, obj_idx, rel_idx))

                elif token.dep_ == "nsubj" and token.head.pos_ == "ADJ":
                    subject_tokens = self._extract_phrase_tokens(token)
                    subject_text = " ".join([t.text for t in subject_tokens])
                    subject_idx = [(subject_tokens[0].i, subject_tokens[-1].i + 1)]

                    adj_token = token.head
                    relation_text = f"is {adj_token.text}"
                    rel_idx = [adj_token.i]

                    obj_tokens = self._extract_subject_complement_tokens(adj_token)
                    if obj_tokens:
                        obj_text = " ".join([t.text for t in obj_tokens])
                        obj_idx = (obj_tokens[0].i, obj_tokens[0].i + 1)
                        triplets.append((subject_text, relation_text, obj_text,
                                       subject_idx, obj_idx, rel_idx))

                elif token.dep_ == "nsubj" and token.head.pos_ == "NOUN":
                    subject_tokens = self._extract_phrase_tokens(token)
                    subject_text = " ".join([t.text for t in subject_tokens])
                    subject_idx = [(subject_tokens[0].i, subject_tokens[-1].i + 1)]

                    noun_token = token.head
                    relation_text = "is"
                    rel_idx = [noun_token.i]

                    obj_tokens = self._extract_phrase_tokens(noun_token)
                    obj_text = " ".join([t.text for t in obj_tokens])
                    obj_idx = (noun_token.i, noun_token.i + 1)

                    triplets.append((subject_text, relation_text, obj_text,
                                   subject_idx, obj_idx, rel_idx))
        except (TypeError, AttributeError, ValueError):
            pass

        return triplets

    def _extract_phrase_tokens(self, token):
        """Extract tokens in a phrase including modifiers."""
        phrase_tokens = [token]

        try:
            for child in token.children:
                if child.dep_ in ["amod", "compound", "det", "poss", "nummod"]:
                    phrase_tokens.append(child)
                elif child.dep_ == "prep" and token.pos_ != "VERB":
                    for subchild in child.children:
                        if subchild.dep_ == "pobj":
                            phrase_tokens.extend([child, subchild])
                            break
        except (TypeError, AttributeError):
            pass

        phrase_tokens.sort(key=lambda x: x.i)
        return phrase_tokens

    def _extract_subject_complement_tokens(self, token):
        """Extract subject complement tokens."""
        try:
            for child in token.children:
                if child.dep_ == "attr" or (child.dep_ == "acomp" and child.pos_ == "ADJ"):
                    return self._extract_phrase_tokens(child)
        except (TypeError, AttributeError):
            pass
        return None

    def _extract_biomedical_relations(self, doc) -> list[Triple]:
        """Extract relations using biomedical entity recognition."""
        triples = []

        try:
            for ent in doc.ents:
                if ent.label_ in ["GENE", "CHEMICAL", "DISEASE", "PROTEIN", "DRUG"]:
                    head = ent.root.head
                    if head.pos_ == "VERB":
                        for other_ent in doc.ents:
                            if other_ent != ent and other_ent.root.head == head:
                                triples.append((
                                    ent.text,
                                    head.text,
                                    other_ent.text,
                                    (ent.start, ent.end),
                                    (other_ent.start, other_ent.end),
                                    (head.i, head.i + 1)
                                ))
        except (TypeError, AttributeError):
            pass

        return triples

    def _resolve_coref_improved(self, doc) -> str:
        """Improved coreference resolution with better detection."""
        if not self.config.use_coref:
            return doc.text
        
        resolved_parts = []
        current_pos = 0
        
        try:
            if hasattr(doc._, 'coref_clusters') and doc._.coref_clusters:
                clusters = sorted(doc._.coref_clusters, key=lambda c: c.mentions[0].start if c.mentions else 0)
                
                for cluster in clusters:
                    if not cluster.mentions:
                        continue
                    
                    main_mention = cluster.main.text
                    
                    if main_mention.lower() in ['he', 'she', 'it', 'they', 'we', 'you']:
                        for mention in cluster.mentions:
                            if mention.text.lower() not in ['he', 'she', 'it', 'they', 'we', 'you']:
                                main_mention = mention.text
                                break
                    
                    for mention in cluster.mentions:
                        if mention.start < current_pos:
                            continue
                        
                        if current_pos < mention.start:
                            resolved_parts.append(doc.text[current_pos:mention.start])
                        
                        if mention.text.lower() != main_mention.lower():
                            resolved_parts.append(main_mention)
                        else:
                            resolved_parts.append(mention.text)
                        
                        current_pos = mention.end
                
                if current_pos < len(doc.text):
                    resolved_parts.append(doc.text[current_pos:])
                
                resolved_text = "".join(resolved_parts)
                if resolved_text and resolved_text != doc.text:
                    return resolved_text
                    
        except (AttributeError, TypeError, ValueError):
            pass
        
        return doc.text

    @staticmethod
    def _covered_token_indices(triples) -> set[int]:
        """Flatten indices from triples into a single set."""
        covered: set[int] = set()

        def _add(item):
            if item is None:
                return
            if isinstance(item, int):
                covered.add(item)
            elif isinstance(item, tuple) and len(item) == 2 and all(isinstance(x, int) for x in item):
                covered.update(range(item[0], item[1]))
            elif isinstance(item, (list, tuple, set)):
                for sub in item:
                    _add(sub)

        for t in triples:
            for field in t[3:6]:
                _add(field)
        return covered

    def _check_completeness(self, triples: list[Triple], sent) -> bool:
        """Enhanced completeness check with content-based analysis."""
        content_indices = set()
        content_words = set()
        
        try:
            for token in sent:
                if not token.is_punct and token.pos_ not in ["PUNCT", "CCONJ", "DET", "PART"]:
                    content_indices.add(token.i)
                    if token.pos_ in ["VERB", "NOUN", "PROPN", "ADJ", "ADV"]:
                        content_words.add(token.text.lower())
        except (TypeError, AttributeError):
            return True
        
        if not content_indices:
            return True
        
        covered_indices = self._covered_token_indices(triples)
        coverage_ratio = len(covered_indices & content_indices) / len(content_indices)
        
        has_verb_relation = False
        for t in triples:
            if t[3] and isinstance(t[3][0], tuple) if t[3] else False:
                try:
                    for idx_range in t[3]:
                        if isinstance(idx_range, tuple):
                            for tok in sent[idx_range[0]:idx_range[1]]:
                                if tok.pos_ == "VERB":
                                    has_verb_relation = True
                                    break
                except (TypeError, AttributeError, IndexError):
                    pass
        
        if coverage_ratio < self.config.coverage_threshold:
            if has_verb_relation:
                return True
            key_words = {"is", "are", "was", "were", "has", "have", "had", "said", "reported"}
            for token in sent:
                if token.text.lower() in key_words and token.i in covered_indices:
                    return True
            return False
        
        return True

    def _filter_redundant_prep_triples(self, triples: list[Triple]) -> list[Triple]:
        """Remove redundant prepositional relation triples."""
        if self.config.deduplication_level == 'strict':
            return triples
            
        filtered = []
        for t in triples:
            try:
                if " to " in t[1]:
                    redundant = False
                    for other in filtered:
                        if (t[0].lower() == other[0].lower() and 
                            t[2].lower() == other[2].lower() and 
                            len(other[1].split()) > len(t[1].split()) and
                            t[1].lower() in other[1].lower()):
                            redundant = True
                            break
                    if not redundant:
                        filtered.append(t)
                else:
                    filtered.append(t)
            except (TypeError, AttributeError):
                filtered.append(t)
        return filtered

    def _process_sentence(self, sent) -> list[Triple]:
        """Process a single sentence with all extractors."""
        triples: list[Triple] = []

        try:
            seen_verbs: set[int] = set()
            for token in sent:
                if token.pos_ in {"VERB", "AUX"} and token.i not in seen_verbs:
                    seen_verbs.add(token.i)
                    for conj in self._expand_conj(token):
                        if conj.i not in seen_verbs:
                            seen_verbs.add(conj.i)
                    triples.extend(self._verb_triples(token))
                    triples.extend(self._advcl_triples(token))
                    triples.extend(self._extract_nested_clauses_recursive(token))
                elif (token.pos_ == "AUX" and token.dep_ == "auxpass") and token.i not in seen_verbs:
                    if token.dep_ == "auxpass" or (token.head.pos_ == "VERB" and token.head.dep_ == "ROOT"):
                        seen_verbs.add(token.head.i)
                        for conj in self._expand_conj(token.head):
                            if conj.i not in seen_verbs:
                                seen_verbs.add(conj.i)
                        triples.extend(self._verb_triples(token.head))
                        triples.extend(self._advcl_triples(token.head))
                        triples.extend(self._extract_nested_clauses_recursive(token.head))
                    else:
                        seen_verbs.add(token.i)
                        for conj in self._expand_conj(token):
                            if conj.i not in seen_verbs:
                                seen_verbs.add(conj.i)
                        triples.extend(self._verb_triples(token))
                        triples.extend(self._advcl_triples(token))
                        triples.extend(self._extract_nested_clauses_recursive(token))
                elif token.pos_ in {"VERB", "AUX"} and token.i in seen_verbs:
                    triples.extend(self._verb_triples(token))
                    triples.extend(self._advcl_triples(token))
                    triples.extend(self._extract_nested_clauses_recursive(token))
                if token.dep_ == "acl":
                    triples.extend(self._acl_triples(token))

            triples.extend(self._copular_triples(sent))
            triples.extend(self._apposition_triples(sent))
            triples.extend(self._extract_complex_appositions(sent))
            triples.extend(self._possessive_triples(sent))
            triples.extend(self._nmod_compound_triples(sent))
            triples.extend(self._attributive_adj_triples(sent))
            triples.extend(self._existential_triples(sent))
            
            if self.config.extract_numerical:
                triples.extend(self._extract_numerical_relations(sent))
            if self.config.extract_quotes:
                triples.extend(self._extract_reported_speech(sent))
            triples.extend(self._extract_relative_clauses(sent))
            triples.extend(self._extract_noun_phrase_relations(sent))
            #triples.extend(self._handle_complex_noun_phrases(sent))

            if self.config.include_context:
                triples.extend(self._extract_with_context(sent))

            #if self.config.use_fallback and not self._check_completeness(triples, sent):
                #triples.extend(self._fallback(sent))
        except (TypeError, AttributeError, ValueError):
            pass

        return triples

    def _process_sentence_with_gap_fill(self, sent) -> list[Triple]:
        """Process sentence with gap-filling fallback."""
        triples = self._process_sentence(sent)

        if self.config.use_fallback and not self._check_completeness(triples, sent):
            triples.extend(self._extract_triple_and_handler(sent))
        
        triples = self._filter_redundant_prep_triples(triples)

        return triples
    
    def _extract_nested_clauses_recursive(self, token, depth=0) -> list[Triple]:
        """Recursively extract triples from nested clauses."""
        triples = []
        
        if depth > self.config.max_recursion_depth:
            return triples
        
        try:
            for child in token.children:
                if child.dep_ in ["relcl", "acl", "advcl", "ccomp", "xcomp"] and child.pos_ == "VERB":
                    rel, rel_idx = self._relation_span(child)
                    subjects = self._get_subjects(child)
                    objects = self._get_objects(child)
                    
                    head_noun = self._relcl_subject(child)
                    if head_noun and not subjects:
                        subjects = [head_noun]
                    
                    if not subjects:
                        subjects = self._get_subjects(token)
                    
                    for subj in subjects:
                        s_text, s_idx = self._clean_subject_span(subj)
                        
                        if objects:
                            for obj in objects:
                                o_text, o_idx = self._subtree_span(obj, exclude_deps=CLAUSAL_LEAK_DEPS)
                                triples.append((s_text, rel, o_text, [s_idx], o_idx, rel_idx))
                        else:
                            for subchild in child.children:
                                if subchild.dep_ == "prep":
                                    for pobj in subchild.children:
                                        if pobj.dep_ == "pobj":
                                            o_text, o_idx = self._subtree_span(pobj, exclude_deps=CLAUSAL_LEAK_DEPS)
                                            triples.append((s_text, f"{rel} {subchild.text}", o_text, [s_idx], o_idx, rel_idx))
                    
                    triples.extend(self._extract_nested_clauses_recursive(child, depth + 1))
        except (TypeError, AttributeError, ValueError):
            pass
        
        return triples

    # ─── Biomedical Extraction ───────────────────────────────────────────────

    def extract_biomedical_triples(self, doc) -> tuple[list[Triple], list]:
        """Extract triples with biomedical entity focus."""
        triples = []
        token_lists = []

        triples.extend(self._extract_biomedical_relations(doc))
        regular_triples, _ = self.extract(doc)
        triples.extend(regular_triples)

        for sent in doc.sents:
            token_lists.append([token.text for token in sent])

        return self._deduplicate(triples), token_lists

    def extract_with_quality(self, doc) -> tuple[list[Triple], list, Dict]:
        """Extract triples with quality scores for each triple."""
        triples, token_lists = self.extract(doc)
        
        quality_info = {
            'total_triples': len(triples),
            'triples_with_scores': []
        }
        
        for t in triples:
            score = self._calculate_quality_score(t, doc)
            quality_info['triples_with_scores'].append({
                'triple': t,
                'score': score
            })
        
        return triples, token_lists, quality_info


# ─── Usage Example ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import spacy
    
    # Load model
    nlp = spacy.load("en_core_web_trf")
    
    # Configure extractor
    config = TripleExtractorConfig()
    config.deduplication_level = 'aggressive'
    config.clean_output = True
    config.use_fallback = True
    
    extractor = TripleExtractor(nlp, config)
    
    # Example text
    text = """Tokyo is the capital city of Japan. The Greater Tokyo Area is the most populous metropolitan area in the world. 
    Chilly Gonzales is a Grammy-winning Canadian musician who resided in Paris, France for several years, and now lives in Cologne, Germany."""
    
    doc = nlp(text)
    triples, tokens = extractor.extract(doc)
    
    print("Extracted Triples:")
    for i, (subj, rel, obj, *_) in enumerate(triples, 1):
        print(f"{i}. {subj} | {rel} | {obj}")