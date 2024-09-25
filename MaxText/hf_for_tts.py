import datasets
import transformers
import grain.python as grain
from input_pipeline import _input_pipeline_utils
import multihost_dataloading
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from functools import partial

import jax
from jax import numpy as jnp
import numpy as np
import dac_jax
import tensorflow as tf
from array_record.python.array_record_module import ArrayRecordWriter
MAX_LENGTH_AUDIO = 40 * 44100   
MAX_LENGTH_TEXT = 5120
GLOBAL_BATCH_SIZE = 32
class HFParseAudioFeatures(grain.MapTransform):
  """Normalize feature keys for HuggingFace input"""
  def map(self, features):
    return {
        "audio": np.asarray(features["audio"]["array"], dtype=np.float32),
        "text": np.asarray(features["text"], dtype=np.int32),
    }   

class PadToMaxLength(grain.MapTransform):

  def map(self, data):
    audio_length = data["audio"].shape[0]
    padded_audio = np.pad(data["audio"],(0,MAX_LENGTH_AUDIO - data["audio"].shape[0]))
    text_length = data["text"].shape[0]
    padded_text = np.pad(data["text"],(0,MAX_LENGTH_TEXT - data["text"].shape[0]))
    return {
        "audio": padded_audio,
        "audio_length":audio_length,
        "text": padded_text,
        "text_length":text_length
    }
if __name__ == "__main__":
    device_mesh = mesh_utils.create_device_mesh((4, 1))
    mesh = Mesh(device_mesh, axis_names=("data", "model")) 
    dataset = datasets.load_dataset(
        "mozilla-foundation/common_voice_17_0",
        name="en",
        split="validated",
        streaming=True,
        
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "fishaudio/fish-speech-1",
        add_bos_token=False,
        add_eos_token=False,
        model_max_length=15000,
        legacy=False,
    
    )
    def prepare_dataset(batch):
        """Function to preprocess the dataset with the .map method"""
        transcription = batch["sentence"]
        
        if transcription.startswith('"') and transcription.endswith('"'):
            # we can remove trailing quotation marks as they do not affect the transcription        
            transcription = transcription[1:-1]
        
        if transcription[-1] not in [".", "?", "!"]:
            # append a full-stop to sentences that do not end in punctuation
            transcription = transcription + "."
        
        batch["sentence"] = transcription
        
        return batch

    dataset = dataset.map(prepare_dataset)
    dataset = dataset.map(
        lambda examples: tokenizer(examples["sentence"]),
        batched=True
    )
    def get_sharding_for_spec(pspec: PartitionSpec) -> NamedSharding:
        """
        Get a NamedSharding for a given PartitionSpec, and the device mesh.
        A NamedSharding is simply a combination of a PartitionSpec and a Mesh instance.
        """
        return NamedSharding(mesh, pspec)
    model, variables = dac_jax.load_model(model_type="44khz")
    x_sharding = get_sharding_for_spec(PartitionSpec("data"))
    @partial(jax.jit, in_shardings=x_sharding,out_shardings=x_sharding)
    def encode_to_codes(x: jnp.ndarray):
        codes, scale = model.apply(
            variables,
            x,
            method="encode",
        )
        return codes, scale
    dataset = dataset.select_columns(["input_ids","audio"]).rename_column("input_ids", "text")
    dataset = _input_pipeline_utils.HFDataSource(dataset,
                                                0,
                                                1,
                                                1,
                                                False,
                                                15000,
                                                "text")
    operations = []
    operations.append(HFParseAudioFeatures())
    operations.append(PadToMaxLength())
    operations.append(grain.Batch(batch_size=GLOBAL_BATCH_SIZE, drop_remainder=True))
    dummy_index_sampler = grain.IndexSampler(
      num_records=len(dataset),
      num_epochs=1,
      shard_options=grain.ShardOptions(
          shard_index=0, shard_count=1, drop_remainder=True
      ),
      shuffle=False,
      seed=0,
    )

    dataloader = grain.DataLoader(
        data_source=dataset,
        operations=operations,
        sampler=dummy_index_sampler,
        worker_count=1,  # only supports one worker for now, more workers results in duplicated data
        worker_buffer_size=1,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=128),
    )
    

    multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, mesh)

    CODEBOOK_PAD_TOKEN_ID = 0
    i = 0
    writer = None
    for item in multihost_gen:
        print(f"round {i}")
        if i%10240 == 0:
            num = i//10240
            if writer is not None:
                writer.close() 
            writer = ArrayRecordWriter(f"/home/fbsdev005/bucket/fish_speech_ds/llm/mozilla_common_voice_train_part_{num}.arrayrecord", 'group_size:1')
        
        #Baatch Length
        semantics, scale = encode_to_codes(jnp.expand_dims(item["audio"],1))
        #Batch Codebook Length 
        semantics = np.asarray(semantics)
        
        #writer = ArrayRecordWriter(f"/home/fbsdev005/bucket/fish_speech_ds/llm/train_part_{num}.arrayrecord", 'group_size:1')
        for k in range(GLOBAL_BATCH_SIZE):
            n_frames = item["audio_length"][k]//512
            text_length = item["text_length"][k]
            #Batch Length -> Length
            text_tokens = item["text"][k][:text_length]
            #Batch Codebook Length  -> Codebook Length 
            semantics_slice = semantics[k][:,:n_frames]

            string_prefix = "<|im_start|>user\n"
            string_suffix = "<|im_end|><|im_start|>assistant\n"

            encoded_prefix = tokenizer.encode(
                string_prefix,
                add_special_tokens=False,
                max_length=10**6,
                truncation=False,
            )

            encoded_suffix = tokenizer.encode(
                string_suffix,
                add_special_tokens=False,
                max_length=10**6,
                truncation=False,
            )

            encoded = encoded_prefix + np.asarray(text_tokens).tolist() + encoded_suffix
            codebook_dim = 9
            semantic_token_id = tokenizer.convert_tokens_to_ids("<|semantic|>")
            semantic_length = semantics_slice.shape[1]#sum([len(i[0]) for i in semantics])
            tokens = (
                encoded
                + [semantic_token_id] * semantic_length
                + tokenizer.convert_tokens_to_ids(["<|im_end|>"])
            )
            prompt_length = len(encoded)
            codes = [[CODEBOOK_PAD_TOKEN_ID] * prompt_length for _ in range(codebook_dim)]
            #for segment in semantics:
            for book_idx, book in zip(range(codebook_dim), semantics_slice):
                for j in book:
                    codes[book_idx].append(int(j) + 1)

            for book in codes:
                book.extend([CODEBOOK_PAD_TOKEN_ID] * 1)
            tokens = [tokens] + codes
            tokens = np.asarray(tokens)
            labels = tokens.copy()
            labels[1:, :prompt_length] = -100
            tokens = tokens[:, :-1]
            labels = labels[:, 1:]
            i+=1
            example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'inputs': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tokens).numpy()])),
                            'targets': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(labels).numpy()])),
                            'prompt_length':tf.train.Feature(
                               int64_list=tf.train.Int64List(value=[prompt_length])
                            )
                            #'speaker':tf.train.Feature(bytes_list=tf.train.BytesList(value=[item["speaker"].encode('utf-8')]))
                        }
                    )
                )
            writer.write(example.SerializeToString())
        #writer.close() 
    