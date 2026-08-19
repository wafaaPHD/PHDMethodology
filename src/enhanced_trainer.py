# enhanced_trainer.py
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from .enhanced_model import EnhancedTriplePredictionModel
from .enhanced_loss import EnhancedTwoHeadedLoss
from .PerformanceMetrics import PerformanceMetrics
from .preprocessing_funcs import load_dataloaders
from .train_funcs import load_state, load_results, evaluate_
from .misc import save_as_pickleResult, load_pickle
import logging
import json
import time
import numpy as np
import pandas as pd

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def train_enhanced_model(args, modelt):
    """
    Train the enhanced model with triple prediction
    """
    if args.fp16:
        from apex import amp
    else:
        amp = None
    
    cuda = torch.cuda.is_available()
    
    # Load data
    train_loader = load_dataloaders(args, modelt)   
    args.ResultPathDataset=args.PathDataset
    args.triple_loss_weight = 0.5  # Weight for triple prediction loss
    train_len = len(train_loader)
    logger.info("Loaded %d pre-training samples." % train_len)
    
    # Initialize model
    if args.model_no == 0:
        from transformers import BertConfig
        config = BertConfig.from_pretrained(args.model_size)
        config.num_relations = args.num_relations  # Add this to args
        net = EnhancedTriplePredictionModel.from_pretrained(
            args.model_size, 
            config=config,
            args=args,
            force_download=False
        )
        model_name = 'BERT'
    
    tokenizer = load_pickle("%s_tokenizer.pkl" % model_name, args)
    net.resize_token_embeddings(len(tokenizer))
    
    if cuda:
        net.cuda()
    
    # Set up optimizer and scheduler
    criterion = EnhancedTwoHeadedLoss(
        lm_ignore_idx=tokenizer.pad_token_id, 
        use_logits=True, 
        normalize=True,
        triple_loss_weight=args.triple_loss_weight if hasattr(args, 'triple_loss_weight') else 0.5
    )
    optimizer = optim.AdamW([{"params": net.parameters(), "lr": args.lr}])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,24,26,30], gamma=0.8)
    
    start_epoch, best_pred, amp_checkpoint = load_state(net, optimizer, scheduler, args, load_best=False)
    
    if args.fp16 and amp is not None:
        logger.info("Using fp16...")
        net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
        if amp_checkpoint is not None:
            amp.load_state_dict(amp_checkpoint)
    
    losses_per_epoch, f1_per_epoch = load_results(args)
    
    logger.info("Starting enhanced training process...")
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    update_size = 100
    # Initialize metrics tracker
    metrics_tracker = PerformanceMetrics()

    # Calculate model complexity before training (add before training loop)
    model_complexity = metrics_tracker.calculate_model_complexity(net, args)
    print(f"Model has {model_complexity['total']:,} parameters")
    print(f"Estimated FLOPs: {model_complexity['flops_total']:,}")
    print("Theoretical memory footprint: {:.2f} MB".format((model_complexity['total'] * 4) / (1024 * 1024)))

    # Initialize triple metrics tracking
    triple_metrics = {
        'relation_precision': [],
        'relation_recall': [],
        'relation_f1': [],
        'relation_auc': [],
        'triple_accuracy': []
    }
    excel_filename = args.ResultPathDataset+'training_metrics.xlsx'
    epoch_logs = []  # Store epoch summary dictionaries
    for epoch in range(start_epoch, args.num_epochs):
        # Start performance tracking for this epoch
        metrics_tracker.start_epoch_tracking()
        start_time = time.time()       
        net.train()
        total_loss = 0.0
        textresult = ''
        losses_per_batch = []
        precision_per_batch = []
        recall_per_batch = []
        f1_per_batch = [] 
        auc_per_batch = []
        Triple_precision_per_batch = []
        Triple_recall_per_batch = []
        Triple_f1_per_batch = []
        Triple_auc_per_batch = []
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_auc = 0.0
        auc_count = 0
        # Triple prediction metrics
        total_triple_acc = 0.0
        total_triple_precision = 0.0
        total_triple_recall = 0.0
        total_triple_f1 = 0.0
        triple_count = 0
        # Training statistics for complexity analysis
        training_stats = {
            'losses': [],
            'rel_prec': [],
            'rel_rec': [],
            'rel_f1': [],
            'rel_auc': [],
            'trip_prec': [],
            'trip_rec': [],
            'trip_f1': [],
            'trip_auc': [],
            'grad_norms': []
            }
        for i, data in enumerate(train_loader, 0):
            # Start batch timing
            batch_start_time = time.time()
            # BUGFIX: the dataset/collate function now also produces a real
            # per-example relation_id (see preprocessing_funcs.py); it used
            # to be dropped entirely, forcing relation_labels to always be
            # None below and silently disabling the whole triple-loss branch.
            x, masked_for_pred, e1_e2_start,blank_labels,triple_targets,_,_,_,_,_,relation_ids_batch = data
            #x, masked_for_pred, e1_e2_start, _, blank_labels, _, _, _, _, triple_targets, relation_ids_batch = data
            masked_for_pred = masked_for_pred[(masked_for_pred != pad_id)]
            if masked_for_pred.shape[0] == 0:
                continue
                
            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()
            
            if cuda:
                x = x.cuda()
                masked_for_pred = masked_for_pred.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                # BUGFIX: was splitting into a (tensor, tensor) tuple here,
                # but the model does `e1_e2_start[:, 0]`, which only works on
                # the original [batch, 2] tensor, not on a tuple. Just move
                # the whole tensor to the GPU and keep its shape.
                e1_e2_start = e1_e2_start.cuda()
                relation_ids_batch = relation_ids_batch.cuda()
            
            # Forward pass with triple predictions
            result=net(
                x, 
                token_type_ids=token_type_ids, 
                attention_mask=attention_mask, 
                e1_e2_start=e1_e2_start,
                return_triple_predictions=True
            )
            lm_logits=result['lm_logits']
            lm_logits = lm_logits[(x == mask_id)]
            
            # Real relation labels for the triple/relation-classification head.
            # -1 entries (padding or examples that couldn't be mapped to a
            # relation id during preprocessing) are ignored by the loss/metrics.
            relation_labels = relation_ids_batch.view(-1)
            
            verbose = (i % update_size) == (update_size - 1)
            triple_predictions=result['triple_predictions']
            blanks_logits=result['blanks_logits']
            loss = criterion(
                lm_logits, 
                blanks_logits, 
                masked_for_pred, 
                blank_labels,
                triple_predictions=triple_predictions,
                relation_labels=relation_labels,
                args=args,
                verbose=verbose
            )
            #loss=result['loss']
            loss = loss / args.gradient_acc_steps
            
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            if args.fp16:
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
            else:
                grad_norm = clip_grad_norm_(net.parameters(), args.max_norm)
            
            if (i % args.gradient_acc_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            # Run evaluation
            (rel_prec, rel_rec, rel_f1, rel_auc), (trip_prec, trip_rec, trip_f1, trip_auc) = evaluate_(
                lm_logits=lm_logits,
                blanks_logits=blanks_logits,
                masked_for_pred=masked_for_pred,
                blank_labels=blank_labels,
                tokenizer=tokenizer,
                triple_predictions=triple_predictions,
                relation_labels=relation_labels,
                triple_targets=triple_targets  # Binary vector [1, 0, 1, 0...] for true/corrupted triples
            )
            #print(f"Relation F1: {rel_f1:.4f} | Relation AUC: {rel_auc:.4f}")
            #print(f"Triple F1:   {trip_f1:.4f} | Triple AUC:   {trip_auc:.4f}")
            total_precision += rel_prec
            total_recall += rel_rec
            total_f1 += rel_f1
            total_triple_precision += trip_prec
            total_triple_recall += trip_rec
            total_triple_f1 += trip_f1
            # Track performance for this batch
            batch_size = x.size(0)
            seq_length = x.size(1)
            batch_metrics = metrics_tracker.track_batch(batch_start_time, batch_size, seq_length, net)
            if not math.isnan(trip_auc):
                total_triple_acc += trip_auc
                triple_count += 1
            if not math.isnan(rel_auc):
                total_auc += rel_auc
                auc_count += 1
            # Record training stats
            training_stats['losses'].append(loss.item())
            training_stats['rel_prec'].append(rel_prec)
            training_stats['rel_rec'].append(rel_rec)
            training_stats['rel_f1'].append(rel_f1)
            training_stats['rel_auc'].append(rel_auc)
            training_stats['trip_prec'].append(trip_prec)
            training_stats['trip_rec'].append(trip_rec)
            training_stats['trip_f1'].append(trip_f1)
            training_stats['trip_auc'].append(trip_auc)
            training_stats['grad_norms'].append(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
            if (i % update_size) == (update_size - 1):
                losses_per_batch.append(args.gradient_acc_steps*total_loss/update_size)
                precision_per_batch.append(total_precision/update_size)
                recall_per_batch.append(total_recall/update_size)
                f1_per_batch.append(total_f1/update_size)
                auc_per_batch.append(total_auc/auc_count if auc_count > 0 else float('nan'))
                 
                Triple_precision_per_batch.append(total_triple_precision/update_size)
                Triple_recall_per_batch.append(total_triple_recall/update_size)
                Triple_f1_per_batch.append(total_triple_f1/update_size)
                Triple_auc_per_batch.append(total_triple_acc/auc_count if auc_count > 0 else float('nan'))
                 
                print('[Epoch: %d, %5d/ %d points] total loss, precision, recall, f1, auc per batch: %.3f, %.3f, %.3f, %.3f, %.3f' %
                      (epoch + 1, (i + 1), train_len, losses_per_batch[-1], precision_per_batch[-1], \
                       recall_per_batch[-1], f1_per_batch[-1], auc_per_batch[-1]))
                print('[Epoch: %d, %5d/ %d points] total loss, precision, recall, f1, auc per batch Triple clasification: %.3f, %.3f, %.3f, %.3f, %.3f' %
                      (epoch + 1, (i + 1), train_len, losses_per_batch[-1], Triple_precision_per_batch[-1], \
                       Triple_recall_per_batch[-1], Triple_f1_per_batch[-1], Triple_auc_per_batch[-1]))


                textresult+='\n'+('[Epoch: %d, %5d/ %d points] total loss, precision, recall, f1, auc per batch: %.3f, %.3f, %.3f, %.3f, %.3f' %
                      (epoch + 1, (i + 1), train_len, losses_per_batch[-1], precision_per_batch[-1], \
                       recall_per_batch[-1], f1_per_batch[-1], auc_per_batch[-1]))
                textresult+='\n'+('[Epoch: %d, %5d/ %d points] total loss, precision, recall, f1, auc per batch Triple clasification: %.3f, %.3f, %.3f, %.3f, %.3f' %
                      (epoch + 1, (i + 1), train_len, losses_per_batch[-1], Triple_precision_per_batch[-1], \
                       Triple_recall_per_batch[-1], Triple_f1_per_batch[-1], Triple_auc_per_batch[-1]))

                with open(args.ResultPathDataset+'result1.txt', 'a') as f:
                      json.dump(textresult, f)
                      f.write("\n")
                with open(args.ResultPathDataset+'result3.txt', 'a') as f:
                      json.dump({'precision': precision_per_batch[-1], 'recall': recall_per_batch[-1], \
                                 'f1': f1_per_batch[-1], 'auc': auc_per_batch[-1],
                                 't_precision': Triple_precision_per_batch[-1], 'T_recall': Triple_recall_per_batch[-1], \
                                 't_f1': Triple_f1_per_batch[-1], 'T_auc': Triple_auc_per_batch[-1]}, f)
                      f.write("\n")
                textresult=''
                
                total_loss = 0.0; total_precision = 0.0; total_recall = 0.0; total_f1 = 0.0; total_auc = 0.0; auc_count = 0
                logger.info("Last batch samples (pos, neg): %d, %d" % ((blank_labels.squeeze() == 1).sum().item(),\
                                                                    (blank_labels.squeeze() == 0).sum().item()))

            if verbose:
                print(f'[Epoch: {epoch+1}, {i+1}/{train_len}] '
                      f'Loss: {losses_per_batch[-1] if losses_per_batch else 0:.3f}, '
                      f'F1: {f1_per_batch[-1] if f1_per_batch else 0:.3f}, AUC: {auc_per_batch[-1] if auc_per_batch else 0:.3f}, '
                      f't_F1: {Triple_f1_per_batch[-1] if Triple_f1_per_batch else 0:.3f}, T-AUC: {Triple_auc_per_batch[-1] if Triple_auc_per_batch else 0:.3f}')
        
        # End of epoch
        scheduler.step()
        
        # Save metrics
        if len(losses_per_batch) > 0:
            avg_loss = sum(losses_per_batch) / len(losses_per_batch)
            avg_f1 = sum(f1_per_batch) / len(f1_per_batch) if f1_per_batch else 0
            
            losses_per_epoch.append(avg_loss)
            f1_per_epoch.append(avg_f1)
        
        # Log results
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} finished in {elapsed:.2f}s")
        print(f"Avg Loss: {losses_per_epoch[-1] if losses_per_epoch else 0:.5f}")
        print(f"Avg F1: {f1_per_epoch[-1] if f1_per_epoch else 0:.5f}")
        # End epoch tracking and get metrics
        epoch_metrics = metrics_tracker.end_epoch_tracking(epoch, args.num_epochs)
        # 2. Compute mean values for the epoch (ignoring NaNs for AUC if any)
        epoch_data = {
            "Epoch": epoch + 1,
            "Loss": np.nanmean(losses_per_batch),
            "Precision": np.nanmean(precision_per_batch),
            "Recall": np.nanmean(recall_per_batch),
            "F1": np.nanmean(f1_per_batch),
            "AUC": np.nanmean(auc_per_batch),
            "Triple_Precision": np.nanmean(Triple_precision_per_batch),
            "Triple_Recall": np.nanmean(Triple_recall_per_batch),
            "Triple_F1": np.nanmean(Triple_f1_per_batch),
            "Triple_AUC": np.nanmean(Triple_auc_per_batch),
        }

        epoch_logs.append(epoch_data)

        # 3. Convert to DataFrame and update/overwrite Excel file
        df = pd.DataFrame(epoch_logs)
        df.to_excel(excel_filename, index=False)
        print(f"Epoch {epoch + 1} metrics updated in {excel_filename}")
    
        print("Epoch finished, took %.2f seconds." % epoch_metrics['epoch_time'])
        print("CPU Memory: %.0f MB" % epoch_metrics['cpu_memory'])
        print("GPU Memory: %.0f MB (peak: %.0f MB)" % (epoch_metrics['gpu_memory']['current'],
                                                    epoch_metrics['gpu_memory']['peak']))
        try:
            print("Losses at Epoch %d: %.7f" % (epoch + 1, losses_per_epoch[-1]))
            print("F1 at Epoch %d: %.7f" % (epoch + 1, f1_per_epoch[-1]))
        except:
            pass
        valid_aucs_epoch = [a for a in auc_per_batch if not math.isnan(a)]
        auc_epoch = sum(valid_aucs_epoch)/len(valid_aucs_epoch) if len(valid_aucs_epoch) > 0 else float('nan')
        print("AUC at Epoch %d: %.7f" % (epoch + 1, auc_epoch))
        # Save complexity analysis every few epochs
        if (epoch + 1) % 5 == 0 or epoch == args.num_epochs - 1:
            complexity_report = metrics_tracker.save_complexity_analysis(
            args, epoch, training_stats
            )
        
        textresult = '\n' + 'Epoch finished, took %.2f seconds.' % epoch_metrics['epoch_time']
        textresult += '\n' + 'CPU Memory: %.0f MB' % epoch_metrics['cpu_memory']
        textresult += '\n' + 'GPU Memory: %.0f MB (peak: %.0f MB)' % (epoch_metrics['gpu_memory']['current'],
                                                                   epoch_metrics['gpu_memory']['peak'])
        try:
            textresult += '\n' + 'Losses at Epoch %d: %.7f' % (epoch + 1, losses_per_epoch[-1])
            textresult += '\n' + 'F1 at Epoch %d: %.7f' % (epoch + 1, f1_per_epoch[-1])
        except:
            pass

        textresult += '\n' + 'AUC at Epoch %d: %.7f' % (epoch + 1, auc_epoch)
    
        if len(precision_per_batch)!=0:
            textresult+='\n'+'Precision at Epoch %d: %.7f' % (epoch + 1, sum(precision_per_batch)/len(precision_per_batch))
        if len(recall_per_batch)!=0:
            textresult+='\n'+'Recall at Epoch %d: %.7f' % (epoch + 1, sum(recall_per_batch)/len(recall_per_batch))

        with open(args.ResultPathDataset+'result2.txt', 'a') as f:
                     json.dump(textresult, f)
                     f.write("\n")
        
        if f1_per_epoch[-1] > best_pred:
            best_pred = f1_per_epoch[-1]
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': f1_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                    'amp': amp.state_dict() if amp is not None else amp
                }, os.path.join(args.ResultPathDataset , "test_model_best_%d.pth.tar" % args.model_no))
        
        if (epoch % 1) == 0:
            save_as_pickleResult("test_losses_per_epoch_%d.pkl" % args.model_no, losses_per_epoch,args)
            save_as_pickleResult("test_f1_per_epoch_%d.pkl" % args.model_no, f1_per_epoch,args)
            # Save checkpoint
            torch.save({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_acc': best_pred,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'amp': amp.state_dict() if amp is not None else amp
            }, os.path.join(args.ResultPathDataset, f"test_checkpoint_{args.model_no}.pth.tar"))
    
    logger.info("Finished Training!")
    return net