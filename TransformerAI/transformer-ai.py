import os
import csv
import pandas as pd
import pretty_midi as pm
import torch
import mido


# Convert MIDI files to a single CSV file
def midi_to_multiple_csv(midi_directory, output_csv):
    max = 0
    num_of_midi = 0
    # Iterate over each MIDI file
    for midi_file in os.listdir(midi_directory):
        num_of_midi = num_of_midi + 1
        # Load the MIDI file
        midi_data = mido.MidiFile(os.path.join(midi_directory, midi_file))

        # Create an empty DataFrame to store the extracted features
        df = pd.DataFrame(columns=['time', 'note', 'velocity'])

        # Extract the desired features from the MIDI data
        message_num = 0
        for message in midi_data:
            if message.type == 'note_on':
                df = pd.concat([df, pd.DataFrame({'time': message.time, 'note': message.note, 'velocity': message.velocity}, index=[len(df)])])
            message_num = message_num + 1
        # Save the extracted features to a CSV file
        df.to_csv(os.path.join(output_csv, os.path.splitext(midi_file.title())[0] + '.csv'), index=False)
        if max < message_num:
            max = message_num

    return max, num_of_midi

def midi_to_single_csv(midi_directory, output_csv):
    num_of_midi = 0

    # Create an empty DataFrame to store the extracted features
    df = pd.DataFrame(columns=['time', 'note', 'velocity'])

    # Iterate over each MIDI file
    for midi_file in os.listdir(midi_directory):
        num_of_midi = num_of_midi + 1
        # Load the MIDI file
        midi_data = mido.MidiFile(os.path.join(midi_directory, midi_file))

        # Extract the desired features from the MIDI data
        message_num = 0
        for message in midi_data:
            if message.type == 'note_on':
                df = pd.concat([df, pd.DataFrame({'time': message.time, 'note': message.note, 'velocity': message.velocity}, index=[len(df)])])
            message_num = message_num + 1
    # Save the extracted features to a CSV file
    df.to_csv(os.path.join(output_csv + 'all_midi_files.csv'), index=False)

    return num_of_midi

midi_dir = r"C:\Users\ydego\Documents\GitHub\MusicTransformerProject\TransformerAI\files_midi"
csv_dir = r"C:\Users\ydego\Documents\GitHub\MusicTransformerProject\TransformerAI\files_csv"
# max_length, num_of_midi = midi_to_multiple_csv(midi_dir, csv_dir)
# num_of_midi = midi_to_single_csv(midi_dir, csv_dir)
max_length = 288
num_of_midi = 70

tensor_list = []
for midi_csv in os.listdir(csv_dir):
    # Read the CSV file containing the extracted features
    df = pd.read_csv(os.path.join(csv_dir, midi_csv.title()))

    # Convert the DataFrame to a tensor
    temp_tensor = torch.tensor(df.values).float()

    # Pad with zeroes
    pad = (0, 0, 0, max_length-temp_tensor.size(dim=0))
    tensor_padded = torch.nn.functional.pad(temp_tensor, pad, "constant", 0)
    tensor_list.append(tensor_padded)

input_tensor = torch.stack(tensor_list, dim=0)

# Build the transformer model
model = torch.nn.Transformer(d_model=num_of_midi, nhead=7, num_encoder_layers=6, num_decoder_layers=6)

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    # Clear the gradients
    model.zero_grad()

    # Forward pass
    output, _ = model(src=input_tensor, tgt=input_tensor)

    # Compute the loss
    loss = loss_fn(output, input_tensor)

    # Backward pass
    loss.backward()

    # Update the parameters
    optimizer.step()

    # Print the loss
    print(f'Epoch {epoch + 1}: Loss = {loss.item()}')

# Generate music using the trained model
output_tensor = model.generate(input_tensor)

# Convert the output tensor to a DataFrame
output_df = pd.DataFrame(output_tensor.detach().numpy(), columns=['feature1', 'feature2', 'feature3'])

# Save the output DataFrame to a CSV file
output_df.to_csv
