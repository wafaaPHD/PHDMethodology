import time
import psutil
import numpy as np
from datetime import datetime
import os
from torch.cuda import memory_stats, memory_allocated, memory_reserved
import torch
class PerformanceMetrics:
    def __init__(self):
        self.epoch_times = []
        self.batch_times = []
        self.gpu_memory_usage = []
        self.cpu_memory_usage = []
        self.flops_per_batch = []
        self.model_params = 0
        self.batch_sizes = []
        self.sequence_lengths = []
        
    def calculate_model_complexity(self, model, input_shape):
        """Calculate theoretical FLOPs and parameter count"""
        from torchinfo import summary
        import torch.nn as nn
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate FLOPs for a forward pass
        # This is a simplified estimation - you may want to use more precise methods
        flops_forward = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                flops_forward += module.in_features * module.out_features
            elif isinstance(module, nn.LSTM):
                flops_forward += 4 * module.hidden_size * (module.input_size + module.hidden_size)
            elif isinstance(module, nn.GRU):
                flops_forward += 3 * module.hidden_size * (module.input_size + module.hidden_size)
            elif isinstance(module, nn.MultiheadAttention):
                # Approximate attention FLOPs
                embed_dim = module.embed_dim
                num_heads = module.num_heads
                flops_forward += 3 * embed_dim * embed_dim  # QKV projections
                flops_forward += embed_dim * embed_dim  # Output projection
                flops_forward += 2 * embed_dim * embed_dim  # Attention scores and softmax
        
        # Backward pass typically requires ~2x FLOPs of forward
        total_flops = flops_forward * 3  # Forward + backward + overhead
        
        self.model_params = {
            'total': total_params,
            'trainable': trainable_params,
            'flops_forward': flops_forward,
            'flops_total': total_flops
        }
        
        return self.model_params
    
    def start_epoch_tracking(self):
        """Initialize tracking for an epoch"""
        self.epoch_start_time = time.time()
        self.epoch_start_memory_cpu = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        if torch.cuda.is_available():
            self.epoch_start_memory_gpu = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            self.epoch_start_memory_gpu = 0
        self.batch_times = []
        self.batch_gpu_memory = []
        self.batch_cpu_memory = []
        
    def track_batch(self, batch_start_time, batch_size, seq_length, model=None):
        """Track performance for a single batch"""
        # Time tracking
        batch_time = time.time() - batch_start_time
        self.batch_times.append(batch_time)
        self.batch_sizes.append(batch_size)
        self.sequence_lengths.append(seq_length)
        
        # Memory tracking
        cpu_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        self.batch_cpu_memory.append(cpu_memory)
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            self.batch_gpu_memory.append(gpu_memory)
            
            # Log GPU memory stats
            memory_stats_dict = torch.cuda.memory_stats()
            if len(self.batch_gpu_memory) > 1:
                memory_allocated_delta = self.batch_gpu_memory[-1] - self.batch_gpu_memory[-2]
            else:
                memory_allocated_delta = gpu_memory - self.epoch_start_memory_gpu
        else:
            self.batch_gpu_memory.append(0)
            memory_allocated_delta = 0
        
        return {
            'batch_time': batch_time,
            'cpu_memory': cpu_memory,
            'gpu_memory': gpu_memory if torch.cuda.is_available() else 0,
            'memory_delta': memory_allocated_delta
        }
    
    def end_epoch_tracking(self, epoch, total_epochs):
        """End epoch tracking and record metrics"""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        # CPU memory
        cpu_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        self.cpu_memory_usage.append(cpu_memory)
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            self.gpu_memory_usage.append({
                'current': gpu_memory,
                'peak': peak_gpu_memory,
                'reserved': torch.cuda.memory_reserved() / 1024 / 1024
            })
            
            # Clear GPU memory stats for next epoch
            torch.cuda.reset_peak_memory_stats()
        else:
            self.gpu_memory_usage.append({
                'current': 0,
                'peak': 0,
                'reserved': 0
            })
        
        return {
            'epoch_time': epoch_time,
            'cpu_memory': cpu_memory,
            'gpu_memory': self.gpu_memory_usage[-1]
        }
    
    def save_complexity_analysis(self, args, epoch, training_stats):
        """Save theoretical and empirical complexity analysis"""
        import json
        from datetime import datetime
        
        # Calculate statistics
        avg_batch_time = np.mean(self.batch_times) if self.batch_times else 0
        std_batch_time = np.std(self.batch_times) if self.batch_times else 0
        avg_cpu_memory = np.mean(self.batch_cpu_memory) if self.batch_cpu_memory else 0
        avg_gpu_memory = np.mean(self.batch_gpu_memory) if self.batch_gpu_memory else 0
        
        # Calculate throughput
        avg_batch_size = np.mean(self.batch_sizes) if self.batch_sizes else 0
        avg_seq_length = np.mean(self.sequence_lengths) if self.sequence_lengths else 0
        tokens_per_second = (avg_batch_size * avg_seq_length) / avg_batch_time if avg_batch_time > 0 else 0
        
        # Model complexity
        model_complexity = self.model_params
        
        # Training time projection
        if self.epoch_times:
            avg_epoch_time = np.mean(self.epoch_times)
            total_estimated_time = avg_epoch_time * (args.num_epochs - epoch - 1)
        else:
            avg_epoch_time = 0
            total_estimated_time = 0
        
        complexity_report = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch + 1,
            'model_complexity': {
                'total_parameters': model_complexity.get('total', 0),
                'trainable_parameters': model_complexity.get('trainable', 0),
                'estimated_flops_forward': model_complexity.get('flops_forward', 0),
                'estimated_flops_total': model_complexity.get('flops_total', 0),
                'memory_footprint_mb': (model_complexity.get('total', 0) * 4) / (1024 * 1024)  # Assume 4 bytes per param
            },
            'empirical_metrics': {
                'epoch_time_seconds': {
                    'current': self.epoch_times[-1] if self.epoch_times else 0,
                    'average': avg_epoch_time,
                    'total_estimated': total_estimated_time
                },
                'batch_time_seconds': {
                    'average': avg_batch_time,
                    'std': std_batch_time,
                    'min': np.min(self.batch_times) if self.batch_times else 0,
                    'max': np.max(self.batch_times) if self.batch_times else 0
                },
                'cpu_memory_mb': {
                    'current': self.cpu_memory_usage if hasattr(self, 'cpu_memory_usage') and self.cpu_memory_usage else 0,
                    'average': avg_cpu_memory,
                    'peak': max(self.batch_cpu_memory) if self.batch_cpu_memory else 0
                },
                'gpu_memory_mb': {
                    'current': self.gpu_memory_usage[-1].get('current', 0) if self.gpu_memory_usage else 0,
                    'peak': self.gpu_memory_usage[-1].get('peak', 0) if self.gpu_memory_usage else 0,
                    'reserved': self.gpu_memory_usage[-1].get('reserved', 0) if self.gpu_memory_usage else 0,
                    'average': avg_gpu_memory
                },
                'throughput': {
                    'tokens_per_second': tokens_per_second,
                    'samples_per_second': avg_batch_size / avg_batch_time if avg_batch_time > 0 else 0,
                    'batches_per_second': 1.0 / avg_batch_time if avg_batch_time > 0 else 0
                }
            },
            'training_stats': training_stats
        }
        
        # Save complexity report
        report_path = os.path.join(args.ResultPathDataset, f'complexity_analysis_epoch_{epoch+1}.json')
        with open(report_path, 'w') as f:
            json.dump(complexity_report, f, indent=2)
        
        # Print formatted summary
        self.print_complexity_summary(complexity_report)
        
        return complexity_report
    
    def print_complexity_summary(self, report):
        """Print a formatted summary of complexity analysis"""
        print("\n" + "="*80)
        print("COMPLEXITY ANALYSIS SUMMARY")
        print("="*80)
        
        print("\n[Model Architecture]")
        print(f"  Total Parameters: {report['model_complexity']['total_parameters']:,}")
        print(f"  Trainable Parameters: {report['model_complexity']['trainable_parameters']:,}")
        print(f"  Estimated FLOPs (Forward): {report['model_complexity']['estimated_flops_forward']:,}")
        print(f"  Estimated FLOPs (Total): {report['model_complexity']['estimated_flops_total']:,}")
        print(f"  Memory Footprint (theoretical): {report['model_complexity']['memory_footprint_mb']:.2f} MB")
        
        print("\n[Training Performance]")
        print(f"  Epoch Time: {report['empirical_metrics']['epoch_time_seconds']['current']:.2f} seconds")
        print(f"  Average Epoch Time: {report['empirical_metrics']['epoch_time_seconds']['average']:.2f} seconds")
        print(f"  Estimated Total Remaining Time: {report['empirical_metrics']['epoch_time_seconds']['total_estimated']:.2f} seconds")
        
        print("\n[Memory Usage]")
        print(f"  CPU Memory: {report['empirical_metrics']['cpu_memory_mb']['current']:.0f} MB (peak: {report['empirical_metrics']['cpu_memory_mb']['peak']:.0f} MB)")
        print(f"  GPU Memory: {report['empirical_metrics']['gpu_memory_mb']['current']:.0f} MB (peak: {report['empirical_metrics']['gpu_memory_mb']['peak']:.0f} MB)")
        print(f"  GPU Memory Reserved: {report['empirical_metrics']['gpu_memory_mb']['reserved']:.0f} MB")
        
        print("\n[Throughput]")
        print(f"  Tokens/second: {report['empirical_metrics']['throughput']['tokens_per_second']:.2f}")
        print(f"  Samples/second: {report['empirical_metrics']['throughput']['samples_per_second']:.2f}")
        print(f"  Batches/second: {report['empirical_metrics']['throughput']['batches_per_second']:.2f}")
        
        print("\n[Batch Statistics]")
        print(f"  Average Batch Time: {report['empirical_metrics']['batch_time_seconds']['average']:.3f} seconds")
        print(f"  Batch Time Std: {report['empirical_metrics']['batch_time_seconds']['std']:.3f} seconds")
        print(f"  Batch Time Range: [{report['empirical_metrics']['batch_time_seconds']['min']:.3f}, {report['empirical_metrics']['batch_time_seconds']['max']:.3f}] seconds")
        
        print("\n[Theoretical vs Empirical Comparison]")
        # Compare theoretical FLOPs with empirical time
        theoretical_flops = report['model_complexity']['estimated_flops_total']
        empirical_time = report['empirical_metrics']['epoch_time_seconds']['current']
        if empirical_time > 0:
            achieved_flops = theoretical_flops / empirical_time
            print(f"  Achieved FLOPs/sec: {achieved_flops:,.2f}")
            print(f"  FLOPs Utilization: {(achieved_flops / (1e12 if achieved_flops > 0 else 1))*100:.2f}% of theoretical peak")
        
        print("\n[Recommendations]")
        # Provide optimization recommendations
        if report['empirical_metrics']['gpu_memory_mb']['peak'] > 0.8 * report['empirical_metrics']['gpu_memory_mb']['reserved']:
            print("   GPU memory usage is high. Consider reducing batch size or using gradient accumulation.")
        if report['empirical_metrics']['throughput']['tokens_per_second'] < 1000:
            print("   Low throughput. Consider using mixed precision training or optimizing data loading.")
        if report['empirical_metrics']['batch_time_seconds']['std'] > 0.5 * report['empirical_metrics']['batch_time_seconds']['average']:
            print("   High variance in batch times. Consider optimizing data loading pipeline.")
        
        print("="*80 + "\n")