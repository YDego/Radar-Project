from mido import MidiFile

mid = MidiFile('MIDI Samples\Mullen - Solace (Midi Kit)\M. Almost there - 140 BPM.mid', clip=True)
print(mid)
for track in mid.tracks:
    print(track)
for msg in mid.tracks[0]:
    print(msg)