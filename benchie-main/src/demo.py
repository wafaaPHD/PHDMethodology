#        <NAME OF THE PROGRAM THIS FILE BELONGS TO>
# 	  
#   File:     demo.py
#   Authors:  Kiril Gashteovski (kiril.gashteovski@neclab.eu) 
#             Mingying Yu (mingying.yu@neclab.eu)
#             Bhushan Kotnis (bhushan.kotnis@neclab.eu)
#             Carolin Lawrence (carolin.lawrence@neclab.eu)
#             Goran Glavaš (goran@informatik.uni-mannheim.de)
#             Mathias Niepert (mathias.niepert@neclab.eu)
# 
# NEC Laboratories Europe GmbH, Copyright (c) 2021, All rights reserved.  
# 
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
# 
#        PROPRIETARY INFORMATION ---  
#
# SOFTWARE LICENSE AGREEMENT
#
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
#
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
#
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor. 
#
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
#
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
#
# COPYRIGHT: The Software is owned by Licensor.  
#
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
#
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
#
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
#
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
#
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
#
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
#
# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.  
#
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
#
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
#
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
#
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.  
#
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
#
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
#
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
#
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
#
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
#
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
#
#       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

from benchie import Benchie
import spacy

# import os
# import pickle
# with open('E:/PHD/20-7-2024ResultPDF/NLP/dataset/datasets/openie4_train.pkl', 'rb') as pkl_file:
#         data = pickle.load(pkl_file)


# file1 = open('E:/PHD/20-7-2024ResultPDF/NLP/dataset/datasets/openie4_train.pkl', "r+")
# file1.seek(0)

# print("Output of Readline function is ")
# print(file1.readline())
# print()

# file1 = open('E:/PHD/20-7-2024ResultPDF/NLP/dataset/datasets/NeuralOpenIE/neural_oie.sent', "r+")
# file1.seek(0)

# print("Output of Readline function is ")
# print(file1.readline())
# print()

nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("merge_entities")
nlp.add_pipe("merge_noun_chunks")

# Load gold annotations to BenchIE
benchie = Benchie()
gold_annotation_file = "data/gold/2_annotators/wire57-annotated(57).txt"
myMethod='extractions/wire/AllMyMethod.txt'
# Load gold annotations to BenchIE
benchie = Benchie()
benchie.load_gold_annotations(filename=gold_annotation_file)
benchie.add_oie_system_extractions(oie_system_name="myMethod", filename=myMethod)

# Compute scores
print('print Similarty\n')
benchie.compute_precision(nlp,match_type="Similarty")
benchie.compute_recall(nlp,match_type="Similarty")
benchie.compute_f1()
benchie.print_scores()


# Load gold annotations to BenchIE
benchie = Benchie()
gold_annotation_file = "data/gold/2_annotators/benchie_gold_annotations_en.txt"
myMethod='extractions/benchie/AllMyMethod.txt'
# Load gold annotations to BenchIE
benchie = Benchie()
benchie.load_gold_annotations(filename=gold_annotation_file)
benchie.add_oie_system_extractions(oie_system_name="myMethod", filename=myMethod)

# Compute scores
print('print Similarty\n')
benchie.compute_precision(nlp,match_type="Similarty")
benchie.compute_recall(nlp,match_type="Similarty")
benchie.compute_f1()
benchie.print_scores()



gold_annotation_file = "extractions/RE/re_oie2016_benchie_format.txt"
myMethod='extractions/RE/aggressiveAllMyMethod.txt'
#aggressiveMymethod='extractions/wire/aggressiveAllMyMethod.txt'
#semanticAllMyMethod='extractions/wire/semanticAllMyMethod.txt'
#graph4nlp_extraction_file2 = "extractions/wire/benchie_gold_annotations_en4nlp.txt"
# nlp is already loaded above and reused here (was previously loaded a second time).

# Load gold annotations to BenchIE
benchie = Benchie()
benchie.load_gold_annotations(filename=gold_annotation_file)

# Add OIE systems extractions
#benchie.add_oie_system_extractions(oie_system_name="aggressiveMymethod", filename=aggressiveMymethod)
#benchie.add_oie_system_extractions(oie_system_name="semanticAllMyMethod", filename=semanticAllMyMethod)
#benchie.add_oie_system_extractions(oie_system_name="graph4nlp_extraction_file2", filename=graph4nlp_extraction_file2)
benchie.add_oie_system_extractions(oie_system_name="myMethod", filename=myMethod)

# Compute scores
print('print Similarty\n')
benchie.compute_precision(nlp,match_type="Similarty")
benchie.compute_recall(nlp,match_type="Similarty")
benchie.compute_f1()
benchie.print_scores()


# Define input files
gold_annotation_file = "data/gold/2_annotators/OLD_wire57-annotated(57).txt"
# Load gold annotations to BenchIE
benchie = Benchie()
benchie.load_gold_annotations(filename=gold_annotation_file)

# Reads extractions by systems
reverb = 'extractions/wire/reverb_wire_benchie_form.txt'
clausie ='extractions/wire/clausie_wire_benchie_form.txt'
minie = 'extractions/wire/minie_wire_benchie_form.txt'
imojie = 'extractions/wire/imojie_wire_benchie_form.txt'
openie6 ='extractions/wire/openie6_wire_benchie_form.txt'
m2oie = 'extractions/wire/m2oie_wire_benchie_form.txt'
compactie = 'extractions/wire/compactie_wire_benchie_form.txt'
benchie.add_oie_system_extractions(oie_system_name="reverb", filename=reverb)
benchie.add_oie_system_extractions(oie_system_name="clausie", filename=clausie)
benchie.add_oie_system_extractions(oie_system_name="minie", filename=minie)
benchie.add_oie_system_extractions(oie_system_name="imojie", filename=imojie)
benchie.add_oie_system_extractions(oie_system_name="openie6", filename=openie6)
benchie.add_oie_system_extractions(oie_system_name="m2oie", filename=m2oie)
benchie.add_oie_system_extractions(oie_system_name="compactie", filename=compactie)

# Compute scores
print('print Similarty\n')
benchie.compute_precision(nlp,match_type="Similarty")
benchie.compute_recall(nlp,match_type="Similarty")
benchie.compute_f1()
benchie.print_scores()







# # Run BenchIE for chinese and German
# gold_annotation_file_zh = "data/gold/benchie_gold_annotations_zh.txt"
# multi2oie_extractions_zh_file = "data/oie_systems_explicit_extractions/m2oie_zh_explicit.txt"
# gold_annotation_file_de = "data/gold/benchie_gold_annotations_de.txt"
# multi2oie_extractions_de_file = "data/oie_systems_explicit_extractions/m2oie_de_explicit.txt"

# benchie_zh = Benchie()
# benchie_zh.load_gold_annotations(filename=gold_annotation_file_zh)
# benchie_zh.add_oie_system_extractions(oie_system_name="multi2oie_zh", filename=multi2oie_extractions_zh_file)
# benchie_zh.compute_precision()
# benchie_zh.compute_recall()
# benchie_zh.compute_f1()
# benchie_zh.print_scores()

# benchie_de = Benchie()
# benchie_de.load_gold_annotations(filename=gold_annotation_file_de)
# benchie_de.add_oie_system_extractions(oie_system_name="multi2oie_de", filename=multi2oie_extractions_de_file)
# benchie_de.compute_precision()
# benchie_de.compute_recall()
# benchie_de.compute_f1()
# benchie_de.print_scores()


gold_annotation_file = "data/gold/2_annotators/benchie_gold_annotations_en.txt"

clausie_extractions_file = "data/oie_systems_explicit_extractions/clausie_explicit.txt"
minie_extractions_file = "data/oie_systems_explicit_extractions/minie_explicit.txt"
stanford_extractions_file = "data/oie_systems_explicit_extractions/stanford_explicit.txt"
openie6_extractions_file = "data/oie_systems_explicit_extractions/openie6_explicit.txt"
roie_t_extractions_file = "data/oie_systems_explicit_extractions/roi_t_explicit.txt"
roie_n_extractions_file = "data/oie_systems_explicit_extractions/roi_n_explicit.txt"
naive_extractions_file = "data/oie_systems_explicit_extractions/naive_oie_explicit.txt"
m2oie_extraction_file = "data/oie_systems_explicit_extractions/m2oie_en_explicit.txt"
graph4nlp_extraction_file2 = "data/oie_systems_explicit_extractions/benchie_gold_annotations_enMyMethod.txt"
mymethod_extraction_file = "data/oie_systems_explicit_extractions/aggressiveAllMyMethod.txt"
compactie_benchie_form = "extractions/benchie/compactie_benchie_form.txt"
imojie_benchie_form = "extractions/benchie/imojie_benchie_form.txt"
reverb_benchie_form = "extractions/benchie/reverb_benchie_form.txt"





# Loading en_core_web_lg is one of the most expensive steps in this script (large model,
# several seconds). It was previously loaded twice (once per gold-annotation section below)
# with identical pipes; load it once here and reuse the same nlp object throughout.

benchie.load_gold_annotations(filename=gold_annotation_file)

# Add OIE systems extractions
benchie.add_oie_system_extractions(oie_system_name="MyMethod", filename=mymethod_extraction_file)
benchie.add_oie_system_extractions(oie_system_name="graph4nlp", filename=graph4nlp_extraction_file2)
benchie.add_oie_system_extractions(oie_system_name="clausie", filename=clausie_extractions_file)
benchie.add_oie_system_extractions(oie_system_name="minie", filename=minie_extractions_file)
benchie.add_oie_system_extractions(oie_system_name="stanford", filename=stanford_extractions_file)
benchie.add_oie_system_extractions(oie_system_name="openie6", filename=openie6_extractions_file)
benchie.add_oie_system_extractions(oie_system_name="roie_t", filename=roie_t_extractions_file)
benchie.add_oie_system_extractions(oie_system_name="roie_n", filename=roie_n_extractions_file)
benchie.add_oie_system_extractions(oie_system_name="naive", filename=naive_extractions_file)
benchie.add_oie_system_extractions(oie_system_name="m2oie_en", filename=m2oie_extraction_file)
benchie.add_oie_system_extractions(oie_system_name="compactie_benchie_form", filename=compactie_benchie_form)
benchie.add_oie_system_extractions(oie_system_name="imojie_benchie_form", filename=imojie_benchie_form)
benchie.add_oie_system_extractions(oie_system_name="reverb_benchie_form", filename=reverb_benchie_form)
 
 


print('print Slote\n')
benchie.compute_precision(nlp)
benchie.compute_recall(nlp)
benchie.compute_f1()
benchie.print_scores()
print('print lexical\n')
benchie.compute_precision(nlp,match_type="lexical")
benchie.compute_recall(nlp,match_type="lexical")
benchie.compute_f1()
benchie.print_scores()
# Compute scores
print('print Similarty\n')
benchie.compute_precision(nlp,match_type="Similarty")
benchie.compute_recall(nlp,match_type="Similarty")
benchie.compute_f1()
benchie.print_scores()
