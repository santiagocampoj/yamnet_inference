import tensorflow as tf
import os
import logging
import pandas as pd
import time
import numpy as np
import subprocess
from tensorboard.plugins import projector
import matplotlib.pyplot as plt


def setup_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    print("\nNum GPUs Available: ", len(physical_devices))
    for device in physical_devices:
        print(device)
        print()
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except RuntimeError as e:
            print(e)
            print()


def save_predictions_to_csv(all_data_subfolder, col_names, subfolder_name, subfolder, model_type, logging, window_size=None, stable_version=None):
    logging.info("")
    if window_size is not None:
        if model_type == "port":
            output_filename = f'Port_Model_{subfolder_name}_w_{window_size}s_{stable_version}.csv'
        else:
            output_filename = f'Urban_Model_{subfolder_name}_w_{window_size}s_{stable_version}.csv'
    else:
        if model_type == "port":
            output_filename = f'Port_Model_{subfolder_name}_{stable_version}.csv'
        else:
            output_filename = f'Urban_Model_{subfolder_name}_{stable_version}.csv'
    
    subfolder = subfolder.replace('3-Medidas', '5-Resultados')
    output_folder = os.path.join(subfolder, 'AI_MODEL', 'Predictions')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_full_path = os.path.join(output_folder, output_filename)

    df_subfolder = pd.DataFrame(all_data_subfolder, columns=col_names)
    # order df by date
    df_subfolder = df_subfolder.sort_values(by='date')
    df_subfolder.to_csv(output_full_path, index=False)
    logging.info(f'Output saved to {output_full_path}')



def save_embeddings_funct(embeddings, subfolder_name, subfolder):
    try:
        logging.info("")
        logging.info("Saving embeddings to tensorboard...")
        
        subfolder = subfolder.replace('3-Medidas', '5-Resultados')
        log_dir = os.path.join(subfolder, 'AI_MODEL', 'Embeddings')
        os.makedirs(log_dir, exist_ok=True)

        # save the embeddings as a variable in a TensorFlow checkpoint
        embedding_var = tf.Variable(embeddings, name='yamnet_embeddings')
        checkpoint = tf.train.Checkpoint(embedding=embedding_var)
        checkpoint_path = checkpoint.save(os.path.join(log_dir, 'embedding.ckpt'))

        # prepare metadata file for TensorBoard embeddings projector
        metadata_file = os.path.join(log_dir, 'metadata.tsv')
        with open(metadata_file, 'w') as metadata_writer:
            for index in range(len(embeddings)):
                metadata_writer.write(f"{index}\n")

        # setup the projector config for visualizing embeddings in TensorBoard
        config = projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = embedding_var.name
        embedding_config.metadata_path = 'metadata.tsv'
        
        # save the config file in the log directory
        projector.visualize_embeddings(log_dir, config)

        logging.info(f"Embeddings and metadata saved in {log_dir}")
    
    except Exception as e:
        logging.warning(f"Error saving embeddings: {e}")


def save_spectrogram_w_funct(spectrogram, scores, yamnet_classes, file_name, sr, start_idx=None, end_idx=None, window_size=None):
    try:
        logging.info("")
        logging.info("Saving spectrogram for window size...")
        
        folder_resultados = file_name.replace('3-Medidas', '5-Resultados')
        filename = file_name.split('\\')[-1]
        folder_resultados = '\\'.join(folder_resultados.split('\\')[:-2])
        folder_resultados = os.path.join(folder_resultados, 'AI_MODEL', 'Spectrograms')
        os.makedirs(folder_resultados, exist_ok=True)
        if not os.path.exists(folder_resultados):
            logging.info(f"Creating folder {folder_resultados}")
        else:
            logging.info(f"Folder {folder_resultados} already exists")

        # convert start_idx and end_idx from samples to seconds for accurate plotting
        start_time = start_idx / sr if start_idx is not None else None
        end_time = end_idx / sr if end_idx is not None else None

        logging.info(f"Spectrogram shape: {spectrogram.shape}")

        # Visualization
        plt.figure(figsize=(12, 10))
        title = f"YAMNet predictions for {filename}"
        if start_time is not None and end_time is not None:
            title += f" from {start_time:.2f} to {end_time:.2f} seconds"
            logging.info(f"Plotting spectrogram from {start_time:.2f} to {end_time:.2f} seconds")
        plt.suptitle(title, fontsize=16)

        # Plot log-mel spectrogram
        logging.info("Plotting spectrogram!!")
        plt.subplot(2, 1, 1)
        plt.imshow(spectrogram.T, aspect='auto', interpolation='nearest', origin='lower')
        plt.colorbar(label='Intensity (dB)')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (microseconds)')
        if window_size is not None:
            plt.xlim([0, window_size * 200])
            logging.info(f"Window size: {window_size}")
        else:
            logging.info("No window size specified. Plotting full spectrogram.")

        #calculate real time x-axis for scores plot
        num_frames = scores.shape[0]

        # scores for top-scoring classes
        mean_scores = np.mean(scores, axis=0)
        top_N = 10
        top_class_indices = np.argsort(mean_scores)[::-1][:top_N]
        plt.subplot(2, 1, 2)
        plt.imshow(scores[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')
        
        if start_time is not None and end_time is not None:
            plt.xticks(np.linspace(0, num_frames - 1, 5), labels=np.round(np.linspace(start_time, end_time, 5), 2))
        else:
            plt.xticks(np.linspace(0, num_frames - 1, 5))

        if window_size is not None:
            plt.xlim([0, num_frames - 1])
            logging.info(f"Window size: {window_size}")
        else:
            logging.info("No window size specified. Plotting full prediction map.")

        yticks = range(0, top_N, 1)
        plt.yticks(yticks, [yamnet_classes[i] for i in yticks])
        plt.ylim(-0.5 + np.array([top_N, 0]))
        plt.tight_layout()
        # plt.show()

        # Save the plot
        if start_idx is not None and end_idx is not None:
            output_filename = filename.replace('.wav', '').replace('.WAV', '') + f'_spectrogram_{start_idx}_{end_idx}.png'
        else:
            output_filename = filename.replace('.wav', '').replace('.WAV', '') + '_spectrogram.png'
        
        output_path = os.path.join(folder_resultados, output_filename)
        plt.savefig(output_path)
        plt.close()
        logging.info(f'Spectrogram saved to {output_path}')
    
    except Exception as e:
        logging.warning(f"Error saving spectrogram: {e}")



def print_top_predictions(file_name, predictions, class_names, top_n=5):
    print(f"\nTop {top_n} predictions for {file_name}:")
    top_indices = np.argsort(predictions)[::-1][:top_n] 
    
    for rank, idx in enumerate(top_indices, start=1):
        class_name = class_names[idx]
        probability = predictions[idx]
        print(f"{rank}. {class_name}: {probability:.4f}")
        time.sleep(1)  


def get_audiofiles(path):
    audio_files = [file for file in os.listdir(path) if file.lower().endswith('.wav')]
    return audio_files


def find_audiomoth_folders(base_path):
    for root, dirs, files in os.walk(base_path):
        if 'AUDIOMOTH' in dirs:
            yield root


def list_git_tags():
    try:
        tags = tags = subprocess.check_output(["git", "tag"]).strip().decode()
        return tags.split('\n')
    except subprocess.CalledProcessError:
        return None
    

def select_tag(tags):
    for i, tag in enumerate(tags):
        logging.info(f"{i}: {tag}")
    choice = int(input("Select the tag to use: "))
    tag_selected = tags[choice]
    # replace "." with "_" to be able to use it as a file name
    tag_selected = tag_selected.replace(".", "_")
    return tag_selected


def get_stable_version():
    tags = list_git_tags()
    # get the latest stable version
    tag_selected = tags[-2]
    logging.info(f"Latest stable version: {tag_selected}")
    # replace "." with "_" to be able to use it as a file name
    tag_selected = tag_selected.replace(".", "_")
    logging.info(f"Latest stable version string: {tag_selected}")
    return tag_selected
