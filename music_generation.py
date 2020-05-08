from music21 import converter, instrument, note, chord, stream
import glob
import numpy as np
import tensorflow as tf
import os
import time
import multiprocessing
from joblib import Parallel,delayed

#lets time our code
t23 = time.time()


#function to read notes, make the vocabulary and sparsely encode the vectors
def get_notes(files_list):
        

        notes = []
        start = 0
        lengths = [0]
        
        #go through each song, add all the notes with start character at beginning and end character at the end
        for file_1 in files_list:
            
            notes.append('0')            
            
            midi = converter.parse(file_1)
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts: 
                notes_to_parse = parts.parts[0].recurse()
            else: 
                notes_to_parse = midi.flat.notes

            
            for element in notes_to_parse:
                #if note then simply append value
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                #if chord add chord start and chord end character    
                elif isinstance(element, chord.Chord):
                    notes.append('00')
                    
                    notes += [str(i.pitch) for i in element.notes ]
                    notes.append('01')
            #append end character        
            notes.append("Z")
            #lengths of songs to track training
            length = len(notes) - start
            lengths.append(length)
            start = len(notes)
            
        #create vocabulary
        vocab = {note:i for i,note in enumerate(np.unique(notes))}
        bocav = {i:note for i,note in enumerate(np.unique(notes))}
        sparse = [vocab[i] for i in notes]

        print(f"NO. of songs: {len(lengths)-1}")
        
        return(notes,vocab,bocav,sparse,lengths)




class Musac():
    


    
    
    def __init__(self,path,window):

        #declare the variables
        self.sliding_window = window
        self.checkpoint_dir = path
        
    #make a nice model
    def make_model(self,vocab):
        #notice absence of softmax and logits=True
        self.model = tf.keras.Sequential([

                    tf.keras.layers.Embedding(len(vocab),144),
                    tf.keras.layers.LSTM(256,return_sequences=True),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LSTM(256),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(len(vocab))
        ])
        self.model.compile(tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']
        )
        self.model.summary()


    def data(self,lengths,sparse):
        X=[]
        y=[]
        print("making dataset")
        for i,length in enumerate(lengths[1:]):   
                
        
                
                song = sparse[lengths[i]:lengths[i]+length]
                

                
                X_song = Parallel()(
                    delayed(lambda start:song[start:start+self.sliding_window])(start) for start in range(length - self.sliding_window))
                y_song = list(song[self.sliding_window:])

                X += X_song
                y += y_song
        print("done")
        return(np.array(X),np.array(y))


    # lets train the model
    def train(self,X,y,epoch):   
        
        
        #just to make sure moi GPU is used 
        with tf.device('/device:GPU:0'):
            
            checkpoint_prefix = os.path.join(self.checkpoint_dir, f"time@{time.time()}")

            checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_prefix,
                    save_weights_only=True)
            
            
            

        
            
            self.model.fit(X,
                           y,
                           epochs=epoch,
                           validation_split=0.2,
                           callbacks=[checkpoint_callback])
            print("training done")      
    
    #And they said being a musician is hard
    def generate(self,songs,bocav):
        #chose the song to start off the prediction
        music_sparse = songs
        generating =True
        predict = []
        node = 0
        count = 0
        print("started generating")
        self.model.reset_states()
        while(generating):

            '''
            in case no. of notes < sliding window length
            We are sampling values, i.e, choosing on the basis of probabilities
            which means the most probable note will not necesarily be chosen to keep diversity
            Also the tf.random.categorical function uses logit values not probablities scores 
            hence there is no softmax activation in function
            '''
            
            if(len(music_sparse)<self.sliding_window):
                predictions = self.model(np.reshape(music_sparse,[1,len(music_sparse)]))
                
                prediction = int(tf.random.categorical(predictions, num_samples=1)[-1,0].numpy())
                predict.append([predictions,prediction])

                
            else:
                predictions = self.model(np.reshape(music_sparse[node:node+self.sliding_window],[1,self.sliding_window]))
                
                prediction = int(tf.random.categorical(predictions, num_samples=1)[-1,0].numpy())
                predict.append([predictions,prediction])
                node += 1

            #in case end token is generated
            
            
            if prediction==(len(bocav)-1):
                generating = False
            else:            
                music_sparse.append(prediction)
            count += 1
            
        print("finished generating")
        #return predicted value, alongside notes
        return [bocav[j] for j in music_sparse],predict
        
    #convert notes to midi objects
    def convert(self,output_notes):
        
        count = 1
        output = []
        offset = 0
        while count<(len(output_notes)-1):
            if output_notes[count] != '0': 
                if output_notes[count] == '00' :
                    searching = True
                    chord_notes= []
                    
                    while searching:
                        
                        count += 1
                        if count <(len(output_notes)-1):
                            if output_notes[count] == '01':
                                searching = False
                            else:
                                chord_notes.append(output_notes[count])
                        else:
                            searching = False

                    if len(chord_notes)>0:
                        
                        new_chord = chord.Chord(chord_notes)
                        new_chord.offset = offset 
                        new_chord.storedInstrument = instrument.Piano()
                        output.append(new_chord)
            
                else:
                    
                    if output_notes[count] != '00' and output_notes[count] != '01': 
                    
                        new_note = note.Note(output_notes[count])
                        new_note.offset = offset
                        new_note.storedInstrument = instrument.Piano()
                        output.append(new_note) 
                offset += 0.5
            count += 1
        return(output)

        
    

    #save the midi file  
    def write(self,path_to_save,output):
        midi_file = stream.Stream(output)
        midi_file.write('midi',fp=path_to_save)
        print("done")

def main(file_path,window,checkpoint,name_1,name_2):
    
    path_to_files = file_path 
    files = os.listdir(path_to_files)
    paths = [os.path.join(path_to_files,i) for i in files]
    np.random.shuffle(paths)
    
    '''
    #load saved data if it exists
    notes = list(np.load('D:\\python\\store\\notes.npy',allow_pickle=True))
    lengths = list(np.load('D:\\python\\store\\lengths.npy',allow_pickle=True))    

    vocab = {note:i for i,note in enumerate(np.unique(notes))}
    bocav = {i:note for i,note in enumerate(np.unique(notes))}
    sparse = [vocab[i] for i in notes]
    '''

    notes,vocab,bocav,sparse,lengths = get_notes(paths)
    '''
    #save data
    np.save('notes',notes)
    np.save('lengths',lengths)
    '''
    

    a = Musac(checkpoint,window)
    a.make_model(vocab)
    '''
    X = np.load('X.npy',allow_pickle=True)
    y = np.load('Y.npy',allow_pickle=True)
    '''
    X,y=a.data(lengths[:-1],sparse)
    '''
    np.save('X',X)
    np.save('Y',y)
    '''
    a.train(X,y,5)

    
    output_notes,_ = a.generate(sparse[-lengths[-1]:-lengths[-1]+100],bocav)
    output = a.convert(output_notes)
    a.write(name_1,output)
    '''
    output_notes = a.generate(sparse[:10],bocav)
    output = a.convert(output_notes)
    a.write(name_2,output)
    '''

    print(f"{(time.time()-t23)/60.0}minutes")

    
        
main("D:\\python\\midi_files",100,'./training_checkpoints2','GG1110.mid','GG11.mid')
