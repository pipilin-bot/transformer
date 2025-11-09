"""
Training script for Transformer
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
import json

from sacrebleu import corpus_bleu

try:
    from .transformer import Transformer
    from .data import get_data_loaders
    from .utils import (
        create_padding_mask,
        create_decoder_mask,
        create_lm_mask,
        count_parameters,
        initialize_weights,
    )
except ImportError:
    from transformer import Transformer
    from data import get_data_loaders
    from utils import (
        create_padding_mask,
        create_decoder_mask,
        create_lm_mask,
        count_parameters,
        initialize_weights,
    )
from transformers import AutoTokenizer


def train_epoch(model, train_loader, optimizer, criterion, device, pad_idx=0, task_type='lm',
               scheduler=None, scheduler_type=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        if task_type == 'seq2seq':
            # Seq2Seq: (src, tgt_input, tgt_output)
            src_ids, tgt_input_ids, tgt_output_ids = batch
            src_ids = src_ids.to(device)
            tgt_input_ids = tgt_input_ids.to(device)
            tgt_output_ids = tgt_output_ids.to(device)
            
            # Create masks
            src_mask = create_padding_mask(src_ids, pad_idx)
            tgt_mask = create_decoder_mask(tgt_input_ids, pad_idx)
            
            # Forward pass
            optimizer.zero_grad()
            logits, _, _, _ = model(src_ids, tgt_input_ids, src_mask, tgt_mask)
            
            # Reshape for loss calculation
            logits = logits.view(-1, logits.size(-1))
            tgt_output_ids = tgt_output_ids.view(-1)
            
            # Calculate loss
            loss = criterion(logits, tgt_output_ids)
        else:
            # Language modeling: (input_ids, target_ids)
            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Create causal mask for language modeling
            mask = create_lm_mask(input_ids, pad_idx)
            
            # Forward pass
            optimizer.zero_grad()
            logits, _ = model(input_ids, mask)
            
            # Reshape for loss calculation
            logits = logits.view(-1, logits.size(-1))
            target_ids = target_ids.view(-1)
            
            # Calculate loss
            loss = criterion(logits, target_ids)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler_type == 'cosine' and scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


def greedy_generate_lm(model, start_tokens, steps, pad_idx, device):
    """Greedy autoregressive generation for language modeling."""
    generated = start_tokens
    for _ in range(steps):
        mask = create_lm_mask(generated, pad_idx)
        logits, _ = model(generated, mask)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if generated.size(1) >= start_tokens.size(1) + steps:
            break
    return generated


def evaluate(model, val_loader, criterion, device, pad_idx=0, task_type='lm',
             tokenizer=None, compute_bleu=False):
    """Evaluate on validation/test set and optionally compute BLEU."""
    model.eval()
    total_loss = 0
    num_batches = 0
    hypotheses = []
    references = []

    if compute_bleu and tokenizer is None:
        raise ValueError("Tokenizer must be provided when compute_bleu is True")

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            if task_type == 'seq2seq':
                # Seq2Seq: (src, tgt_input, tgt_output)
                src_ids, tgt_input_ids, tgt_output_ids = batch
                src_ids = src_ids.to(device)
                tgt_input_ids = tgt_input_ids.to(device)
                tgt_output_ids = tgt_output_ids.to(device)

                # Create masks
                src_mask = create_padding_mask(src_ids, pad_idx)
                tgt_mask = create_decoder_mask(tgt_input_ids, pad_idx)

                # Forward pass
                logits, _, _, _ = model(src_ids, tgt_input_ids, src_mask, tgt_mask)

                if compute_bleu:
                    pred_ids = logits.argmax(dim=-1)
                    for pred_seq, tgt_seq in zip(pred_ids, tgt_output_ids):
                        tgt_tokens = tgt_seq[tgt_seq != pad_idx]
                        if tgt_tokens.numel() == 0:
                            continue
                        pred_tokens = pred_seq[:tgt_tokens.size(0)]
                        ref_text = tokenizer.decode(tgt_tokens, skip_special_tokens=True).strip()
                        hyp_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
                        references.append(ref_text if ref_text else " ")
                        hypotheses.append(hyp_text if hyp_text else " ")

                # Reshape for loss calculation
                logits = logits.view(-1, logits.size(-1))
                tgt_output_ids = tgt_output_ids.view(-1)

                # Calculate loss
                loss = criterion(logits, tgt_output_ids)
            else:
                # Language modeling: (input_ids, target_ids)
                input_ids, target_ids = batch
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                target_matrix = target_ids

                # Create causal mask
                mask = create_lm_mask(input_ids, pad_idx)

                # Forward pass
                logits, _ = model(input_ids, mask)

                # Reshape for loss calculation
                logits = logits.view(-1, logits.size(-1))
                target_flat = target_ids.view(-1)

                # Calculate loss
                loss = criterion(logits, target_flat)

                if compute_bleu:
                    batch_size = input_ids.size(0)
                    start_tokens = input_ids[:, :1]
                    generated = greedy_generate_lm(
                        model, start_tokens, target_matrix.size(1), pad_idx, device
                    )
                    pred_sequences = generated[:, 1:1 + target_matrix.size(1)]
                    for pred_seq, tgt_seq in zip(pred_sequences, target_matrix):
                        tgt_tokens = tgt_seq[tgt_seq != pad_idx]
                        if tgt_tokens.numel() == 0:
                            continue
                        pred_trim = pred_seq[:tgt_tokens.size(0)]
                        # Remove padding/eos tokens
                        pred_trim = pred_trim[pred_trim != pad_idx]
                        ref_text = tokenizer.decode(tgt_tokens, skip_special_tokens=True).strip()
                        hyp_text = tokenizer.decode(pred_trim, skip_special_tokens=True).strip()
                        references.append(ref_text if ref_text else " ")
                        hypotheses.append(hyp_text if hyp_text else " ")

            total_loss += loss.item()
            num_batches += 1

    bleu_score = None
    if compute_bleu and len(hypotheses) > 0:
        bleu_score = corpus_bleu(hypotheses, [references]).score

    return total_loss / num_batches, bleu_score


def calculate_perplexity(loss):
    """Calculate perplexity from loss"""
    return math.exp(loss)


def main():
    parser = argparse.ArgumentParser(description='Train Transformer for EN→DE translation')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=512, help='Feed-forward dimension')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--use_learned_pos', action='store_true', help='Use learned positional encoding')
    parser.add_argument('--tokenizer', type=str, default='Helsinki-NLP/opus-mt-en-de', help='Tokenizer to use')
    parser.add_argument('--disable_bleu', action='store_true', help='Disable BLEU computation')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate using the saved checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint for evaluation')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to local IWSLT2017 data directory')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw'], help='Optimizer type')
    parser.add_argument('--lr_scheduler', type=str, default='plateau',
                       choices=['none', 'plateau', 'cosine'],
                       help='Learning rate scheduler')
    parser.add_argument('--warmup_steps', type=int, default=0,
                       help='Warmup steps for cosine scheduler')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[OK] Using GPU: {gpu_name}")
        print(f"     GPU Memory: {gpu_memory:.2f} GB")
        print(f"     CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print("[WARNING] Using CPU for training, this will be very slow!")
        print("          Current PyTorch version: " + torch.__version__)
        if '+cpu' in torch.__version__:
            print("          Detected CPU version of PyTorch, need to install GPU version")
            print("          Install command: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("          If you have a GPU, please check if CUDA drivers are correctly installed")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_idx = tokenizer.pad_token_id
    
    compute_bleu = not args.disable_bleu and args.eval_only

    # Load data
    print("Loading local IWSLT2017 dataset...")
    train_loader, val_loader, test_loader = get_data_loaders(
        tokenizer, args.batch_size, args.max_len, data_dir=args.data_dir
    )
    vocab_size = tokenizer.vocab_size
    src_vocab_size = tgt_vocab_size = vocab_size
    task_type = 'seq2seq'
    print(f"Vocabulary size: {vocab_size}")

    # Create model
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
        use_learned_pos=args.use_learned_pos
    ).to(device)
    
    # Initialize weights (skip when evaluating only)
    if not args.eval_only:
        initialize_weights(model)

    # Count parameters
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = None
    scheduler_type = args.lr_scheduler
    total_training_steps = args.num_epochs * len(train_loader)
    if scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
    elif scheduler_type == 'cosine':
        warmup_steps = max(0, args.warmup_steps)

        def lr_lambda(current_step: int):
            if total_training_steps <= 0:
                return 1.0
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # TensorBoard writer
    writer = None if args.eval_only else SummaryWriter(os.path.join(args.save_dir, 'logs'))
    
    # Prepare config info
    config = {k: v for k, v in vars(args).items() if k not in ['disable_bleu', 'eval_only', 'checkpoint_path']}
    config['compute_bleu'] = compute_bleu
    config['num_params'] = num_params
    config['vocab_size'] = vocab_size
    config['task_type'] = task_type
    config['dataset'] = 'iwslt2017_local'
    if not args.eval_only:
        with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    if args.eval_only:
        checkpoint_path = args.checkpoint_path or os.path.join(args.save_dir, 'best_model.pt')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")

        # Evaluate on validation and test sets
        val_loss, val_bleu = evaluate(
            model, val_loader, criterion, device, pad_idx, task_type,
            tokenizer=tokenizer if compute_bleu else None,
            compute_bleu=compute_bleu
        )
        val_ppl = calculate_perplexity(val_loss)
        print(f"Validation Loss: {val_loss:.4f}, Validation PPL: {val_ppl:.2f}")
        if val_bleu is not None:
            print(f"Validation BLEU: {val_bleu:.2f}")

        print("\nEvaluating on test set...")
        test_loss, test_bleu = evaluate(
            model, test_loader, criterion, device, pad_idx, task_type,
            tokenizer=tokenizer if compute_bleu else None,
            compute_bleu=compute_bleu
        )
        test_ppl = calculate_perplexity(test_loss)
        print(f"Test Loss: {test_loss:.4f}, Test PPL: {test_ppl:.2f}")
        if test_bleu is not None:
            print(f"Test BLEU: {test_bleu:.2f}")

        # Update results file
        results_path = os.path.join(args.save_dir, 'results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
        else:
            results = {}
        results.update({
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'val_bleu': val_bleu,
            'test_loss': test_loss,
            'test_ppl': test_ppl,
            'test_bleu': test_bleu
        })
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print("Evaluation completed!")
        return

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_bleus = [] if compute_bleu else []
    
    print("Starting training...")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, pad_idx, task_type,
            scheduler if scheduler_type == 'cosine' else None,
            scheduler_type=scheduler_type
        )
        train_losses.append(train_loss)
        train_ppl = calculate_perplexity(train_loss)
        
        # Validate
        val_loss, val_bleu = evaluate(
            model, val_loader, criterion, device, pad_idx, task_type,
            tokenizer=tokenizer if compute_bleu else None,
            compute_bleu=compute_bleu
        )
        val_losses.append(val_loss)
        val_ppl = calculate_perplexity(val_loss)
        if compute_bleu:
            val_bleus.append(val_bleu)
        
        # Learning rate scheduling
        if scheduler_type == 'plateau' and scheduler is not None:
            scheduler.step(val_loss)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Perplexity/Train', train_ppl, epoch)
        writer.add_scalar('Perplexity/Validation', val_ppl, epoch)
        if compute_bleu and val_bleu is not None:
            writer.add_scalar('BLEU/Validation', val_bleu, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f}")
        if compute_bleu and val_bleu is not None:
            print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}, Val BLEU: {val_bleu:.2f}")
        else:
            print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_bleu': val_bleu,
                'config': config
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"Saved best model (Val Loss: {val_loss:.4f})")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_bleu = evaluate(
        model, test_loader, criterion, device, pad_idx, task_type,
        tokenizer=tokenizer if compute_bleu else None,
        compute_bleu=compute_bleu
    )
    test_ppl = calculate_perplexity(test_loss)
    if compute_bleu and test_bleu is not None:
        print(f"Test Loss: {test_loss:.4f}, Test PPL: {test_ppl:.2f}, Test BLEU: {test_bleu:.2f}")
    else:
        print(f"Test Loss: {test_loss:.4f}, Test PPL: {test_ppl:.2f}")
    
    # Save final results
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'test_ppl': test_ppl,
        'best_val_loss': best_val_loss
    }
    if compute_bleu:
        results['val_bleus'] = val_bleus
        results['test_bleu'] = test_bleu
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    if writer is not None:
        writer.close()
    print("Training completed!")


if __name__ == '__main__':
    main()

