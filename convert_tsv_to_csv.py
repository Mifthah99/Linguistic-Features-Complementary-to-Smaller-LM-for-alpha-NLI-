import csv


input_path = '/Users/manu/Desktop/Stuttgurt/alphanli/tsv/dev.tsv'  
output_path = '/Users/manu/Desktop/Stuttgurt/alphanli/processed_alphanli.csv'

# Open the TSV file
with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8', newline='') as outfile:
    tsv_reader = csv.reader(infile, delimiter='\t')
    csv_writer = csv.writer(outfile)

    # Write header
    csv_writer.writerow(['s1', 'h', 'is_h1', 'label'])

    for row in tsv_reader:
        if len(row) < 6:
            continue  # Skip bad rows

        story_id, obs1, obs2, hyp1, hyp2, label = row
        label = int(label.strip())

        # First (obs1, hyp1)
        csv_writer.writerow([obs1, hyp1, 1, 1 if label == 1 else 0])
        
        # Second (obs1, hyp2)
        csv_writer.writerow([obs1, hyp2, 0, 1 if label == 2 else 0])

print(f"Done! Saved processed data to {output_path}")
