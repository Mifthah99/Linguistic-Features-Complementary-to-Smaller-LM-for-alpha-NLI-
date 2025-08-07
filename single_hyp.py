import torch
# import torch.distributed as dist # Not directly used for single GPU, can remove
# import torch.nn.parallel as ddp # Not directly used for single GPU, can remove
from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.distributed import DistributedSampler # Not directly used for single GPU, can remove
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import os
import json # Import json for df_from_json

def save_checkpoint(model, optimizer, scaler, scheduler, epoch, val_accuracy, filepath):
    """
    Saves a training checkpoint to a single file.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_accuracy': val_accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath} (Epoch: {epoch}, Val Accuracy: {val_accuracy:.4f})")

def load_checkpoint(model, optimizer, scaler, scheduler, filepath, device):
    """
    Loads a training checkpoint from a single file to resume training.
    """
    start_epoch = 0
    best_val_accuracy = -1.0

    if not os.path.isfile(filepath):
        print(f"No checkpoint found at '{filepath}'. Starting training from scratch.")
        return start_epoch, best_val_accuracy

    print(f"Loading checkpoint from '{filepath}'...")
    try:
        checkpoint = torch.load(filepath, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print("Warning: Scheduler state not found in checkpoint or scheduler is None. Ignoring scheduler loading.")

        start_epoch = checkpoint['epoch'] + 1 # Resume from the next epoch
        best_val_accuracy = checkpoint['val_accuracy']

        print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}, with best validation accuracy {best_val_accuracy:.4f}.")
        return start_epoch, best_val_accuracy

    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting training from scratch.")
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Removed potentially corrupted checkpoint: {filepath}")
        return 0, -1.0



class HypothesisDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256): # Removed target_max_length here as it's handled differently
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.encoded_data = []
        
        # Ensure tokenizer has a pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # Note: Model embeddings resize handled in HypothesisSelector __init__

        print("Pre-tokenizing dataset...")
        for idx in tqdm(range(len(self.data)), desc="Pre-tokenizing"):
            row = self.data.iloc[idx]
            
            # --- For hyp1 ---
            input_text_hyp1 = (
                f"Given these observations:\n"
                f"1: {row['obs1']}\n"
                f"2: {row['obs2']}\n\n"
                f"does this hypothesis align with observations?\n"
                f"hypothesis: {row['hyp1']}\n"
                f"Answer with either '0' or '1'." # The model will generate after this
            )
            
            # Determine the target label for hyp1
            target_label_hyp1 = '1' if str(row['label']) == '1' else '0'

            # Tokenize the input prompt
            encoded_prompt_hyp1 = self.tokenizer(
                input_text_hyp1,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Tokenize the target label ('0' or '1')
            encoded_label_hyp1 = self.tokenizer(
                target_label_hyp1,
                add_special_tokens=False, # Do not add special tokens like BOS/EOS here
                return_tensors='pt'
            )

            # Combine input_ids and labels for Causal LM training
            # The labels for the prompt part should be -100
            # The label for the target token (0 or 1) should be its actual token ID
            # Pad the target label to match the max_length of the prompt + max_new_tokens
            
            # Calculate the actual sequence length of the prompt (without padding)
            prompt_len_hyp1 = encoded_prompt_hyp1['input_ids'].shape[1]
            
            # Create a full label tensor for the combined sequence
            # The total sequence length will be max_length (from prompt) + length of generated label
            # Ensure the target_max_length is at least 1 for the '0' or '1'
            full_labels_hyp1 = torch.full(
                (1, prompt_len_hyp1 + encoded_label_hyp1['input_ids'].shape[1]), 
                -100, # Initialize all with -100
                dtype=torch.long
            )

            # Place the actual label token ID at the end of the sequence
            full_labels_hyp1[0, prompt_len_hyp1:] = encoded_label_hyp1['input_ids'][0]

            # Concatenate input_ids and the target token for training
            # This forms the sequence: [prompt_tokens] + [target_token_ID]
            # The model will learn to predict the target_token_ID after the prompt
            combined_input_ids_hyp1 = torch.cat([
                encoded_prompt_hyp1['input_ids'], 
                encoded_label_hyp1['input_ids']
            ], dim=-1)
            
            # Adjust attention mask for the combined input
            combined_attention_mask_hyp1 = torch.cat([
                encoded_prompt_hyp1['attention_mask'],
                torch.ones_like(encoded_label_hyp1['input_ids']) # Mask for the target token
            ], dim=-1)

            # Pad the combined_input_ids and attention_mask to max_length + target_max_length if needed
            # Or ensure they fit within your model's maximum context length.
            # For simplicity, let's truncate if it exceeds max_length + a small buffer
            total_max_len = self.max_length + 2 # Allow for prompt + '0'/'1'
            if combined_input_ids_hyp1.shape[1] > total_max_len:
                combined_input_ids_hyp1 = combined_input_ids_hyp1[:, :total_max_len]
                combined_attention_mask_hyp1 = combined_attention_mask_hyp1[:, :total_max_len]
                full_labels_hyp1 = full_labels_hyp1[:, :total_max_len]
            elif combined_input_ids_hyp1.shape[1] < total_max_len:
                pad_len = total_max_len - combined_input_ids_hyp1.shape[1]
                combined_input_ids_hyp1 = torch.cat([
                    combined_input_ids_hyp1, 
                    torch.full((1, pad_len), self.tokenizer.pad_token_id, dtype=torch.long)
                ], dim=-1)
                combined_attention_mask_hyp1 = torch.cat([
                    combined_attention_mask_hyp1, 
                    torch.zeros((1, pad_len), dtype=torch.long)
                ], dim=-1)
                full_labels_hyp1 = torch.cat([
                    full_labels_hyp1, 
                    torch.full((1, pad_len), -100, dtype=torch.long)
                ], dim=-1)


            self.encoded_data.append({
                'input_ids': combined_input_ids_hyp1.squeeze(0),
                'attention_mask': combined_attention_mask_hyp1.squeeze(0),
                'labels': full_labels_hyp1.squeeze(0)
            })

            # --- For hyp2 (similar logic) ---
            input_text_hyp2 = (
                f"Given these observations:\n"
                f"1: {row['obs1']}\n"
                f"2: {row['obs2']}\n\n"
                f"does this hypothesis align with observations?\n"
                f"hypothesis: {row['hyp2']}\n"
                f"Answer with either '0' or '1'."
            )
            
            target_label_hyp2 = '1' if str(row['label']) == '2' else '0' # Label '2' for original data means hyp2 is true

            encoded_prompt_hyp2 = self.tokenizer(
                input_text_hyp2,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            encoded_label_hyp2 = self.tokenizer(
                target_label_hyp2,
                add_special_tokens=False,
                return_tensors='pt'
            )

            prompt_len_hyp2 = encoded_prompt_hyp2['input_ids'].shape[1]
            
            full_labels_hyp2 = torch.full(
                (1, prompt_len_hyp2 + encoded_label_hyp2['input_ids'].shape[1]), 
                -100, 
                dtype=torch.long
            )
            full_labels_hyp2[0, prompt_len_hyp2:] = encoded_label_hyp2['input_ids'][0]

            combined_input_ids_hyp2 = torch.cat([
                encoded_prompt_hyp2['input_ids'], 
                encoded_label_hyp2['input_ids']
            ], dim=-1)
            
            combined_attention_mask_hyp2 = torch.cat([
                encoded_prompt_hyp2['attention_mask'],
                torch.ones_like(encoded_label_hyp2['input_ids'])
            ], dim=-1)

            if combined_input_ids_hyp2.shape[1] > total_max_len:
                combined_input_ids_hyp2 = combined_input_ids_hyp2[:, :total_max_len]
                combined_attention_mask_hyp2 = combined_attention_mask_hyp2[:, :total_max_len]
                full_labels_hyp2 = full_labels_hyp2[:, :total_max_len]
            elif combined_input_ids_hyp2.shape[1] < total_max_len:
                pad_len = total_max_len - combined_input_ids_hyp2.shape[1]
                combined_input_ids_hyp2 = torch.cat([
                    combined_input_ids_hyp2, 
                    torch.full((1, pad_len), self.tokenizer.pad_token_id, dtype=torch.long)
                ], dim=-1)
                combined_attention_mask_hyp2 = torch.cat([
                    combined_attention_mask_hyp2, 
                    torch.zeros((1, pad_len), dtype=torch.long)
                ], dim=-1)
                full_labels_hyp2 = torch.cat([
                    full_labels_hyp2, 
                    torch.full((1, pad_len), -100, dtype=torch.long)
                ], dim=-1)


            self.encoded_data.append({
                'input_ids': combined_input_ids_hyp2.squeeze(0),
                'attention_mask': combined_attention_mask_hyp2.squeeze(0),
                'labels': full_labels_hyp2.squeeze(0)
            })
            
        print("Pre-tokenization complete.")

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]
class HypothesisSelector:
    def __init__(self, model_name='google/flan-t5-base', device=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device:
            self.device = device # Allow overriding device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.model.gradient_checkpointing_enable() 
        self.scaler = GradScaler()
        # self.model = torch.compile(self.model) # Uncomment if you want to use torch.compile

    def create_dataloaders(self, train_df, val_df, test_df, batch_size=16, num_workers=0):
        train_dataset = HypothesisDataset(train_df, self.tokenizer)
        val_dataset = HypothesisDataset(val_df, self.tokenizer)
        test_dataset = HypothesisDataset(test_df, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
        return train_loader, val_loader, test_loader

    def train(self, train_loader, val_loader, epochs=5, learning_rate=3e-5, accumulation_steps=1):
        self.model.to(self.device)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        num_training_steps = (len(train_loader) // accumulation_steps) * epochs
        if len(train_loader) % accumulation_steps != 0: 
            num_training_steps += epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=1,
            num_training_steps=num_training_steps
        )

        scaler = self.scaler
        best_accuracy = 0.0
        start_epoch = 0
        checkpoint_path = 'last_checkpoint4.pth'
        best_model_path = 'best_checkpoint4.pth'

        # Load checkpoint if exists
        start_epoch, best_accuracy = load_checkpoint(self.model, optimizer, scaler, scheduler, checkpoint_path, self.device)

        for epoch in range(start_epoch, epochs):
            self.model.train()
            total_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')

            optimizer.zero_grad() # Initialize gradients to zero for the first batch

            for i, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)

                with autocast(dtype=torch.bfloat16): # Ensure your GPU supports bfloat16 for best performance
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False
                    )
                    loss = outputs.loss / accumulation_steps

                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader): # Apply step at end of epoch if batches don't align
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps
                progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})

            avg_loss = total_loss / len(train_loader)
            val_accuracy = self.evaluate(val_loader)
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")

            # Save the last checkpoint always
            save_checkpoint(self.model, optimizer, scaler, scheduler, epoch, val_accuracy, checkpoint_path)

            # Save best checkpoint
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                save_checkpoint(self.model, optimizer, scaler, scheduler, epoch, val_accuracy, best_model_path)
                # Removed redundant torch.save(self.model.state_dict(), best_model_path)

        print(f"Training complete. Best validation accuracy: {best_accuracy:.4f}")

    def evaluate(self, dataloader):
        self.model.eval()
        predictions, actuals = [], []
        
        progress_bar = tqdm(dataloader, desc='Evaluating')

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'] 

                # Generate a small number of new tokens (e.g., 1 or 2)
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=2, # Only generate 1 or 2 tokens to get '0' or '1'
                    num_beams=1,
                    do_sample=False, # Take the most likely token
                    early_stopping=True,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id # Or self.tokenizer.pad_token_id if different
                )

                # Decode only the newly generated part
                # outputs has shape (batch_size, input_length + generated_length)
                generated_tokens = outputs[:, input_ids.shape[1]:] 
                preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                preds = [p.strip().lower() for p in preds] # Strip whitespace

                labels_copy = labels.clone()
                labels_copy[labels_copy == -100] = self.tokenizer.pad_token_id 
                targets = self.tokenizer.batch_decode(labels_copy, skip_special_tokens=True)
                targets = [t.strip().lower() for t in targets]

                for p, t in zip(preds, targets):
                    if p == '1':
                        predictions.append('1')
                    elif p == '0': # Assume '0' is the other valid output
                        predictions.append('0')
                    else:
                        predictions.append('unknown') # For cases where it predicts something else
                    actuals.append(t) 

        return accuracy_score(actuals, predictions)

    def predict(self, obs1, obs2, hyp_text): # Changed to take one hypothesis at a time for clarity
        self.model.eval() 
        input_text = (
            f"Given these observations:\n"
            f"1: {obs1}\n"
            f"2: {obs2}\n\n"
            f"Does this hypothesis align with observations?\n" # Corrected typo 'allign' to 'align'
            f"hypothesis: {hyp_text}\n" # Takes a single hypothesis
            f"Answer with either '0' or '1'. "
        )

        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, padding=True).to(self.device, non_blocking=True)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=2, # Generate only 1 or 2 new tokens
                num_beams=1,
                do_sample=False, # Take the most likely token
                early_stopping=True,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id # Or self.tokenizer.pad_token_id
            )

        # Decode only the newly generated part
        generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
        prediction_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip().lower()

        if prediction_text == '1':
            return 1, prediction_text
        elif prediction_text == '0':
            return 0, prediction_text
        else:
            return -1, prediction_text # Indicate an unexpected output

def df_from_json(input_path, label_path):
    import pandas as pd
    import json
    with open(input_path) as f:
        data = f.readlines()
    with open(label_path) as f:
        labels = f.readlines()
    
    datap = []
    for i in data:
        datap.append(json.loads(i))
    d = pd.DataFrame(datap)
    d.reset_index(inplace=True) # Use inplace=True to modify DataFrame directly
    d['Index'] = d.index
    y = [i.strip() for i in labels]
    d['label'] = y
    return d

def train_model(selector, train_loader, val_loader, test_loader, epochs=5, batch_size=16, num_workers=0,model_save_path='hypothesis_selector_model.pth'):
    """
    Trains the HypothesisSelector model.
    """
    selector.train(train_loader, val_loader, epochs=epochs)
    print('Training complete. Evaluating on test set...')
    # Save the model after training
    # The best model is already saved by save_checkpoint in the train method.
    # If you still want to save the final model state (not necessarily the best), you can do so:
    # torch.save(selector.model.state_dict(), model_save_path) 
    print(f"Model training process includes saving best and last checkpoints.")

if __name__ == "__main__":
    arg_list = os.sys.argv[1:]
    train_data = 'train.json'
    train_labels = 'train_labels.lst'

    test_data = "test.json"
    test_labels = 'test_labels.lst'
    dev_data = 'dev.jsonl'
    dev_labels = 'dev-labels.lst'
    
    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_df = df_from_json(train_data, train_labels)
    test_df = df_from_json(test_data, test_labels)
    dev_df = df_from_json(dev_data, dev_labels)
    print(f"Train DataFrame shape: {train_df.shape}")
    print(f"Test DataFrame shape: {test_df.shape}")
    print(f"Dev DataFrame shape: {dev_df.shape}")

    selector = HypothesisSelector(model_name='google/flan-t5-base', device=device) # Pass device to selector
    
    epochs = 2
    num_workers_to_use = 4 
    batch_size = 1
    
    train_loader, val_loader, test_loader = selector.create_dataloaders(
        train_df, dev_df, test_df, batch_size=batch_size, num_workers=num_workers_to_use)

    if len(arg_list) > 0 and (arg_list[0] == 'train' or arg_list[0] == 'train_model' or arg_list[0] == "t"):
        print("Training the model...")
        train_model(selector, train_loader, val_loader, test_loader, epochs=epochs, batch_size=batch_size, num_workers=num_workers_to_use)
    else:    
        print("Skipping training. Using pre-trained model for predictions. To train pass 'train' or 't' as first argument.")
        print("Will make use of pretrained model at 'best_checkpoint4.pth' if available.")
        
        # Load the best pre-trained model for evaluation and prediction if not training
        model_path = 'best_checkpoint4.pth' # Use the best checkpoint for inference
        if len(arg_list) > 1:
            model_path = arg_list[1] # Allow user to specify a different model path

        if os.path.exists(model_path):
            # When loading only state_dict for inference, optimizer, scaler, scheduler are not needed
            try:
                checkpoint = torch.load(model_path, map_location=selector.device)
                selector.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Pre-trained model loaded from {model_path} (from checkpoint).")
            except Exception as e:
                print(f"Error loading full checkpoint from {model_path}: {e}. Trying to load just state_dict.")
                try:
                    selector.model.load_state_dict(torch.load(model_path, map_location=selector.device))
                    print(f"Pre-trained model state_dict loaded from {model_path}.")
                except Exception as e_sd:
                    print(f"Error loading state_dict from {model_path}: {e_sd}. Starting with base model.")
        else:
            print(f"No pre-trained model found at {model_path}. Please check the path or train the model first. USING BASE PHI-2 MODEL.")
            
    # Prediction examples
    obs1 = "The streets are wet"
    obs2 = "People are carrying umbrellas"
    hyp1 = "It rained recently"
    hyp2 = "The streets were cleaned"

    pred_hyp1, text_hyp1 = selector.predict(obs1, obs2, hyp1)
    print(f"Hypothesis 1 ('{hyp1}'): Predicted: {pred_hyp1} (Output: '{text_hyp1}')")

    pred_hyp2, text_hyp2 = selector.predict(obs1, obs2, hyp2)
    print(f"Hypothesis 2 ('{hyp2}'): Predicted: {pred_hyp2} (Output: '{text_hyp2}')")

    obs1_new = "The birds are singing"
    obs2_new = "The sun is shining brightly"
    hyp1_new = "It's a beautiful morning"
    hyp2_new = "It's about to storm"

    pred_new_hyp1, text_new_hyp1 = selector.predict(obs1_new, obs2_new, hyp1_new)
    print(f"Hypothesis 1 ('{hyp1_new}'): Predicted: {pred_new_hyp1} (Output: '{text_new_hyp1}')")
    
    pred_new_hyp2, text_new_hyp2 = selector.predict(obs1_new, obs2_new, hyp2_new)
    print(f"Hypothesis 2 ('{hyp2_new}'): Predicted: {pred_new_hyp2} (Output: '{text_new_hyp2}')")
    
    DEV = selector.evaluate(val_loader)
    E = selector.evaluate(test_loader)  
    print(f"Dev set evaluation: {DEV:.4f}")
    print(f"Test set evaluation: {E:.4f}")

    with open('result.txt','a') as f:
        f.write(f"no of epochs={epochs} base microsoft/phi-2 \nDEV RESULT:{DEV}\nTEST RESULT:{E}\n")