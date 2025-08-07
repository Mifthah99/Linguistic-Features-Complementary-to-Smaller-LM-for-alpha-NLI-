import torch
import torch.distributed as dist # Keep for DDP context, even if not used directly for single GPU
import torch.nn.parallel as ddp
from torch.utils.data import Dataset, DataLoader, DistributedSampler # Keep DistributedSampler for context
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import os 
def save_checkpoint(model, optimizer, scaler, scheduler, epoch, val_accuracy, filepath):

    # Saves a training checkpoint to a single file.
  
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
    def __init__(self, dataframe, tokenizer, max_length=256, target_max_length=2):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_max_length = target_max_length

        # --- Pre-tokenization happens here ---
        self.encoded_data = []
       
        print("Pre-tokenizing dataset...")
        for idx in tqdm(range(len(self.data)), desc="Pre-tokenizing"):
            row = self.data.iloc[idx]

            input_text = (
                f"Given these observations:\n"
                f"1: {row['obs1']}\n"
                f"2: {row['obs2']}\n\n"
                f"Which hypothesis is more likely?\n"
                f"1: {row['hyp1']}\n"
                f"2: {row['hyp2']}\n"
                f"Answer with either '1' or '2'."
            )
            target_text =str(row['label'])

            inputs = self.tokenizer(
                input_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            targets = self.tokenizer(
                target_text,
                max_length=self.target_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            labels=targets['input_ids'].squeeze(0)
            labels[labels == self.tokenizer.pad_token_id] = -100 
            self.encoded_data.append({
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': labels
            })
        print("Pre-tokenization complete.")
        # --- End of Pre-tokenization ---

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

class HypothesisSelector:
    def __init__(self, model_name='google/flan-t5-base', device=None): # Removed local_rank as we are focusing on single GPU
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.gradient_checkpointing_enable() # Keep for memory efficiency
        # self.model = torch.compile(self.model) # Compile the model for potential performance improvements
        
        self.scaler = GradScaler()
        # self.model= torch.compile(self.model) # Compile the model for potential performance improvements

    def create_dataloaders(self, train_df, val_df, test_df, batch_size=16, num_workers=0): # Added num_workers parameter
        train_dataset = HypothesisDataset(train_df, self.tokenizer)
        val_dataset = HypothesisDataset(val_df, self.tokenizer)
        test_dataset = HypothesisDataset(test_df, self.tokenizer)

        # Added num_workers here
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
        return train_loader, val_loader, test_loader

    def train(self, train_loader, val_loader, epochs=5, learning_rate=3e-5, accumulation_steps=5):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        num_training_steps = (len(train_loader) // accumulation_steps) * epochs
        if len(train_loader) % accumulation_steps != 0: 
            num_training_steps += epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=5,
            num_training_steps=num_training_steps
        )

        scaler = self.scaler
        best_accuracy = 0.0
        start_epoch = 0
        checkpoint_path = 'last_checkpoint.pth'
        best_model_path = 'best_checkpoint.pth'

        # Load checkpoint if exists
        if os.path.exists(checkpoint_path):
            start_epoch, best_accuracy = load_checkpoint(self.model, optimizer, scaler, scheduler, checkpoint_path, self.device)

        for epoch in range(start_epoch, epochs):
            self.model.train()
            total_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')

            for i, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)

                with autocast(dtype=torch.bfloat16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False
                    )
                    loss = outputs.loss / accumulation_steps

                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps
                progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})

            if (len(train_loader) % accumulation_steps != 0) and (i + 1) % accumulation_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader)
            val_accuracy = self.evaluate(val_loader)
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")

            # Save the last checkpoint always
            save_checkpoint(self.model, optimizer, scaler, scheduler, epoch, val_accuracy, checkpoint_path)

            # Save best checkpoint
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                save_checkpoint(self.model, optimizer, scaler, scheduler, epoch, val_accuracy, best_model_path)
                torch.save(self.model.state_dict(), 'best_model.pth')  # optional if you want raw state_dict too

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

                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=2,
                    num_beams=1,
                    early_stopping=True,
                    use_cache=True 
                )

                # Using batch_decode for potentially faster decoding
                preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                preds = [p.lower() for p in preds]

                labels_copy = labels.clone()
                labels_copy[labels_copy == -100] = self.tokenizer.pad_token_id # Handle -100 padding token
                targets = self.tokenizer.batch_decode(labels_copy, skip_special_tokens=True)
                targets = [t.lower().strip() for t in targets]

                for p, t in zip(preds, targets):
                    if '1' in p:
                        predictions.append('1')
                    elif '2' in p:
                        predictions.append('2')
                    else:
                        predictions.append('unknown') 
                    actuals.append(t) # t is already stripped

        return accuracy_score(actuals, predictions)

    def predict(self, obs1, obs2, hyp1, hyp2):
        self.model.eval() 
        input_text = (
            f"Given these observations:\n"
            f"1: {obs1}\n"
            f"2: {obs2}\n\n"
            f"Which hypothesis is more likely?\n"
            f"hyp1: {hyp1}\n"
            f"hyp2: {hyp2}\n"
            f"Answer with either '1' or '2'."
        )

        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, padding=True).to(self.device, non_blocking=True)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=8,
                num_beams=1,
                early_stopping=True,
                use_cache=True 
            )

        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
        if '1' in prediction:
            return 1, prediction
        elif '2' in prediction:
            return 2, prediction
        else:
            return 0, prediction 

def df_from_json(input,label):
    import pandas as pd,json
    with open(input) as f:
        data=f.readlines()
    with open(label)as f:
        labels=f.readlines()
    
    datap=[]
    for i in data:
        datap.append(json.loads(i))
    d=pd.DataFrame(datap)
    d.reset_index()
    d['Index']=d.index
    y=[i.strip()for i in labels]
    d['label']=y
    return d

def train(selector, train_loader, val_loader, test_loader, epochs=5, batch_size=16, num_workers=0,model_save_path='hypothesis_selector_model.pth'):
    """
    Trains the HypothesisSelector model.
    """
    selector.train(train_loader, val_loader, epochs=epochs)
    print('Training complete. Evaluating on test set...')
    # Save the model after training
    torch.save (selector.model.state_dict(), model_save_path)
    print(f"Model saved as {model_save_path}")

    
if __name__ == "__main__":
    arg_list=os.sys.argv[1:]  # Get command line arguments if needed
    train_data='train.json'
    train_labels='train_labels.lst'

    test_data="test.json"
    test_labels='test_labels.lst'
    dev_data='dev.jsonl'
    dev_labels='dev-labels.lst'
    torch.device('cuda:0')
    train_df=df_from_json(train_data,train_labels)
    test_df=df_from_json(test_data,test_labels)
    dev_df=df_from_json(dev_data,dev_labels)
    print(f"Train DataFrame shape: {train_df.shape}")
    print(f"Test DataFrame shape: {test_df.shape}")
    print(f"Dev DataFrame shape: {dev_df.shape}")

    selector = HypothesisSelector(model_name='google/flan-t5-base')
    # torch.load(selector.model.state_dict(), 'hypothesis_selector_model_10.pth')
    epochs=10
    num_workers_to_use = 4 
    batch_size=512
    train_loader, val_loader, test_loader = selector.create_dataloaders(
        train_df, dev_df, test_df, batch_size=batch_size, num_workers=num_workers_to_use)
    if len(arg_list) > 0:
        if arg_list[0]=='train'or arg_list[0]=='train_model' or arg_list[0]=="t":
            print("Training the model...")
            train(selector, train_loader, val_loader, test_loader, epochs=10, batch_size=batch_size, num_workers=num_workers_to_use)
    else:    
        print("Skipping training. Using pre-trained model for predictions.to train pass train  or t as first argument")
        print("will make use of pretrained model at 'hypothesis_selector_model_100.pth' unless new pth file given in arguments")
        # Load the pre-trained model if available
        model_path = 'hypo_model_base_10.pth'
        if len(arg_list) > 1:
            model_path = arg_list[1]
        if os.path.exists(model_path):
            selector.model.load_state_dict(torch.load(model_path, map_location=selector.device))
            print(f"Pre-trained model loaded from {model_path}")
        else:
            print(f"No pre-trained model found at {model_path}. Please check the path or train the model first. USING BASE FLANT5 SMALL MODEL")
            # Set the model to evaluation mode
    # Prediction example
    obs1 = "The streets are wet"
    obs2 = "People are carrying umbrellas"
    hyp1 = "It rained recently"
    hyp2 = "The streets were cleaned"

    pred, text = selector.predict(obs1, obs2, hyp1, hyp2)
    print(f"Predicted hypothesis: {pred} ({text})")

    obs1_new = "The birds are singing"
    obs2_new = "The sun is shining brightly"
    hyp1_new = "It's a beautiful morning"
    hyp2_new = "It's about to storm"

    pred_new, text_new = selector.predict(obs1_new, obs2_new, hyp1_new, hyp2_new)
    print(f"Predicted hypothesis: {pred_new} ({text_new})")
    
    DEV=selector.evaluate(val_loader)

    E=selector.evaluate(test_loader)  
    print(f"Dev set evaluation :\n {DEV:.4f}")
    print(f"Test set evaluation :\n {E:.4f}")

    # Evaluate on test set
    with open('result.txt','a') as f:
        f.write(f"no of epochs={epochs} base T5 \nDEV RESULT:{DEV}\nTEST RESULT{E}\n")