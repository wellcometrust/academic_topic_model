import warnings

import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from torch import multiprocessing
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=UserWarning)
transformers.logging.set_verbosity_error()


class EmbeddingModel:
    """Model and methods for generating sentence embeddings.

    Supports transformer models hosted on HuggingFace Hub.

    By default inference is parralelised per batch over all avalable CUDA GPU
    devices.

    """

    def __init__(self, tokenizer_name, model_name, parallelise=True):
        """Initialise model and parameters.

        Args:
            tokenizer_name(str): HuggingFace Hub tokenizer name.
            model_name(str): HuggingFace Hub model name.
            parallelise(bool): Parallelise over all available GPUs.

        Attributes:.
            device(str): GPU device used, can be CUDA or MPS.
            gpu_count(int): Number of available GPU devices.
            tokenizer(transformers.PreTrainedTokenize): A pre trained
            tokenizer.
            model(transformers.PreTrainedModel): Pretrainned transformer
            model.

        """
        self.device = self.identify_device()

        if self.device == "CUDA":
            torch.cuda.empty_cache()

        transformers.set_seed(101)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)

        self.model = transformers.AutoModel.from_pretrained(model_name)
        self.gpu_count = self.get_device_count(self.device)

        if self.gpu_count > 1 and parallelise:
            self.gpu_parallelism = True
            multiprocessing.set_start_method("spawn")

            print(f"Inference will be parallelised on {self.gpu_count} CUDA GPUs")

            # Set up queues for cross process communication.
            self.input_queue = multiprocessing.Queue()
            self.output_queue = multiprocessing.Queue()

        else:
            self.gpu_parallelism = False
            print(f"Inference will be run on a single {self.device} GPU")

    @staticmethod
    def identify_device():
        """Identify available GPU device.

        Rasises error if only CPU available.

        Returns:
            str: GPU device type (CUDA/MPS).

        """
        if torch.cuda.is_available():
            device = "CUDA"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            raise RuntimeError("Compatible GPU or drivers not found.")

        print(f"PyTorch will initialize using {device}\n")

        return device

    @staticmethod
    def get_device_count(device):
        """Get count of available GPU devices.

        Args:
            device(str): Device type.

        Returns:
            int: Count of avialable GPU devices.

        """
        if device == "CUDA":
            gpu_count = torch.cuda.device_count()
        else:
            gpu_count = 1

        return gpu_count

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        """Performs mean pooling on token embedding sequence.

        Args:
            token_embeddings: Token embeddings.
            attenetion_mask: Attention masking.

        Returns:
            Mean pooled sentence embeddings.

        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        pooled_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return pooled_embeddings

    def generate_embeddings(self, batch_chunk, model, cuda_device_id):
        """Run inference and generate sentence embeddings.

        Note: Batch size is the per GPU batch size when run with parallelism.

        Default batch size optimised for BERT models run on GPUs with 24GB RAM.

        Args:
            batch_chunk(list): Text tokens to embed.
            model: Model on device on which to perform inference.
            cuda_device_id(int): Rank of GPU device to be used by process.

        Returns:
            pd.Series: Pandas series containing sentence embeddings as rows.

        """
        transformers.set_seed(101)

        inputs = batch_chunk.to(cuda_device_id)

        with torch.no_grad():
            embeddings = model(**inputs)[0]

        pooled_embeddings = self.mean_pooling(embeddings, inputs["attention_mask"])

        pooled_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)
        pooled_embeddings = pooled_embeddings.cpu().detach().tolist()

        return pooled_embeddings

    def inference_process(self, cuda_device_id):
        """Process for inference on chunks from input queue.

        Results are placed in the output queue.

        Args:
            cuda_device_id(int): Id of GPU device to be used by process.

        """
        device = f"cuda:{cuda_device_id}"
        model = self.model.to(device)
        model.eval()

        while True:
            batch_chunk = self.input_queue.get()

            if batch_chunk == "KILL":
                break

            batch_id = batch_chunk[0]
            batch_data = batch_chunk[1]

            sentence_embeddings = self.generate_embeddings(
                batch_chunk=batch_data, model=model, cuda_device_id=device
            )

            output = (batch_id, sentence_embeddings)
            self.output_queue.put(output)

    def launch_processes(self):
        """Launch a process for each available GPU.

        Returns:
            list: List of processes.

        """
        processes = []
        for gpu_rank in range(self.gpu_count):
            process = multiprocessing.Process(
                target=self.inference_process, args=(gpu_rank,)
            )

            process.start()
            processes.append(process)

        return processes

    def kill_processes(self):
        """Kill running processes.

        Kills running processes by placing a semaphore in queue.

        """
        for _ in range(self.gpu_count):
            self.input_queue.put("KILL")

    def run_multi_inference(self, input_data, batch_size, auto_launch_and_kill=False):
        """Run model inference in parallel.

        Args:
            input_data(pd.Series, list): Input raw data texts.
            batch_size(int): Batch size to use for inference.

        Returns:
            pd.Series: Sentence embeddings.

        """
        n_rows = len(input_data)

        # Break input into batches.
        n_batches = 0
        for i in range(0, n_rows, batch_size):
            n_batches += 1

            tokens = self.tokenizer(
                input_data[i : i + batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )

            batch_mapping = (i, tokens)
            self.input_queue.put(batch_mapping)

        # Wait until input queue is processed.
        if auto_launch_and_kill:
            self.launch_processes()

        batched_embeddings = []
        while True:
            output = self.output_queue.get()
            batched_embeddings.append(output)

            print(
                f"proportion processed: {len(batched_embeddings)/n_batches*100:.2f}%",
                end="\r",
            )
            if len(batched_embeddings) == n_batches:
                break

        # Re-order embedding outputs.
        batched_embeddings.sort()

        sentence_embeddings = []
        for embedding in batched_embeddings:
            sentence_embeddings += embedding[1]

        sentence_embeddings = pd.Series(sentence_embeddings)

        if auto_launch_and_kill:
            self.kill_processes()
        return sentence_embeddings

    def run_single_inference(self, input_data, batch_size):
        """Run inference single threaded.

        Args:
            input_data(pd.Series, list): Input raw data texts.
            batch_size(int): Batch size to use for inference.

        Returns:
            pd.Series: Sentence embeddings.

        """
        device = self.device.lower()
        model = self.model.to(device)
        model.eval()

        sentence_embeddings = []
        for i in tqdm(range(0, len(input_data), batch_size)):
            tokens = self.tokenizer(
                input_data[i : i + batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )

            embeddings = self.generate_embeddings(
                batch_chunk=tokens, model=model, cuda_device_id=device
            )

            sentence_embeddings += embeddings

        return pd.Series(sentence_embeddings)

    def run_inference(self, input_data, batch_size=400, auto_launch_and_kill=False):
        """Run model inference on text and return sentence embeddings.

        Args:
            input_data(list): Input raw data texts.
            batch_size(int): Batch size to use for inference.

        Returns:
            pd.Series: Sentence embeddings.

        """
        if isinstance(input_data, pd.Series):
            input_data = input_data.to_list()

        if self.gpu_parallelism:
            try:
                embeddings = self.run_multi_inference(
                    input_data, batch_size, auto_launch_and_kill=auto_launch_and_kill
                )
            except KeyboardInterrupt:
                for process in multiprocessing.active_children():
                    process.terminate()

                raise KeyboardInterrupt

        else:
            embeddings = self.run_single_inference(input_data, batch_size)

        return embeddings
