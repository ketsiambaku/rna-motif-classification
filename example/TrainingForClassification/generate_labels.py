import os
import pandas as pd

def generate_csv(base_folder, output_csv):
    data = []
    for label in os.listdir(base_folder):
        label_path = os.path.join(base_folder, label)
        if not os.path.isdir(label_path):
            continue
        
        files = [f.replace('.mrc', '').replace('.pdb', '') 
                for f in os.listdir(label_path) if f.endswith('.mrc')]
        
        for filename in set(files):
            data.append({'filename': filename, 'label': label})
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Generated {output_csv} with {len(df)} entries")
    print(f"Label distribution:\n{df['label'].value_counts()}")

# Generate training CSV
generate_csv('../../dataset', 'rna_labels.csv')
