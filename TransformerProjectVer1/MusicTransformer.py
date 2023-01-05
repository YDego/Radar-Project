import os
import pandas as pd
import torch
import torch.nn as nn
import mido


# Convert MIDI files to multiple CSV files
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
                df = pd.concat([df,
                                pd.DataFrame({'time': message.time, 'note': message.note, 'velocity': message.velocity},
                                             index=[len(df)])])
            message_num = message_num + 1
        # Save the extracted features to a CSV file
        df.to_csv(os.path.join(output_csv, os.path.splitext(midi_file.title())[0] + '.csv'), index=False)
        if max < message_num:
            max = message_num

    print("midi files are converted to csv")
    return max, num_of_midi

# Convert MIDI files to a single CSV file
def midi_to_single_csv(midi_directory, output_csv, csv_filename):
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
                df = pd.concat([df,
                                pd.DataFrame({'time': message.time, 'note': message.note, 'velocity': message.velocity},
                                             index=[len(df)])])
            message_num = message_num + 1
    # Save the extracted features to a CSV file
    df.to_csv(os.path.join(output_csv, csv_filename), index=False)

    print(csv_filename + " file was created, including all midi files")

# Embedding the data
def embedding_data(input_data, d_model):

    # Define the embedding dimensions and the number of unique tokens
    embedding_dim = d_model
    num_tokens = 1000

    # Initialize the embedding layer
    embedding = nn.Embedding(num_tokens, embedding_dim)

    # Perform the embedding
    embedded_data = embedding(input_data.long())

    print("Embedding the data to size: " + str(embedded_data.shape))  # Output: torch.Size([2, 3, 512])

    return embedded_data

# The transformer class
class MusicTransformer(nn.Module):
    def __init__(self, d_model, nhead, nlayers):
        super(MusicTransformer, self).__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout=0.1),
                                             nlayers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, nhead, d_model * 4, dropout=0.1),
                                             nlayers)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, tgt):
        output = self.decoder(tgt, self.encoder(src))
        output = self.norm(self.linear(output))
        return output, None

    def train_model(self, embedded_input, num_epochs):
        # Define the loss function
        loss_fn = torch.nn.CrossEntropyLoss()

        # Define the optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

        # Train the model
        for epoch in range(num_epochs):
            # Clear the gradients
            self.zero_grad()

            # Forward pass
            output, _ = self(src=embedded_input, tgt=embedded_input)

            # Compute the loss
            loss = loss_fn(output, embedded_input)

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Print the loss
            print(f'Epoch {epoch + 1}: Loss = {loss.item()}')



# The code starts here:

csv_filename = 'all_midi_files.csv'
midi_dir = r"C:\Users\ydego\Documents\GitHub\MusicTransformerProject\files_midi"
csv_dir = r"C:\Users\ydego\Documents\GitHub\MusicTransformerProject\files_csv"

# for multiple:     max_length, num_of_midi = midi_to_multiple_csv(midi_dir, csv_dir)
# for single:       midi_to_single_csv(midi_dir, csv_dir)
midi_to_single_csv(midi_dir, csv_dir, csv_filename)

# Read the CSV file containing the extracted features
df = pd.read_csv(os.path.join(csv_dir, csv_filename))

# Convert the DataFrame to a tensor
input_tensor = torch.tensor(df.values).float()

d_model = 128  # 512 originally
nhead = 8
num_layers = 6
num_epochs = 10

# Embedding the data
embedded_input = embedding_data(input_tensor, d_model)

# Build the transformer model
print("Building the transformer now, hold on")
transformer = MusicTransformer(d_model, nhead, num_layers)

# Select device for running the project
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer = transformer.to(device)
embedded_input = embedded_input.to(device)

# Train
print("Training the model")
transformer.train_model(embedded_input, num_epochs)

# Generate music using the trained model
output_tensor = transformer.generate(input_tensor)

# Convert the output tensor to a DataFrame
output_df = pd.DataFrame(output_tensor.detach().numpy(), columns=['time', 'note', 'velocity'])

# Save the output DataFrame to a CSV file
output_df.to_csv()

# Create a track to hold the data
track = mido.MidiTrack()

# Iterate over the rows in the DataFrame and add the data to the track
for index, row in output_df.iterrows():
    time = row['time']
    note = row['note']
    velocity = row['velocity']
    message = mido.Message('note_on', time=time, note=note, velocity=velocity)
    track.append(message)

# Save the MIDI file
mid = mido.MidiFile()
mid.tracks.append(track)
mid.save(mid, 'output.mid')






