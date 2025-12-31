import os
import csv
import numpy as np
from typing import Iterator, Optional, Sequence, Union, Dict, List
from tqdm import tqdm

class StreamingCSVSamples:
    """
    A streaming row sampler for a CSV that:
      - yields one row at a time (as np.ndarray)
      - keeps an internal pointer
      - auto-rewinds to the beginning when EOF is reached
      - supports skiprows + usecols
    """

    def __init__(
        self,
        csv_filename: str,
        *,
        skiprows: int = 0,
        usecols: Optional[Union[Sequence[int], range]] = None,
        has_header: bool = True,
        dtype=float,
    ):
        self.csv_filename = csv_filename
        self.skiprows = skiprows
        self.usecols = list(usecols) if usecols is not None else None
        self.has_header = has_header
        self.dtype = dtype

        self._fh = None
        self._reader = None
        self._open_and_position()

    def _open_and_position(self) -> None:
        """(Re)open the file and move the pointer to the first data row."""
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass

        self._fh = open(self.csv_filename, "r", newline="")
        self._reader = csv.reader(self._fh)

        # Skip preamble rows
        for _ in range(self.skiprows):
            next(self._reader, None)

        # Optionally skip the header row
        if self.has_header:
            next(self._reader, None)

    def _read_next_data_row(self) -> Optional[np.ndarray]:
        """
        Read the next row; return None if EOF.
        Skips empty/non-numeric rows safely.
        """
        while True:
            row = next(self._reader, None)
            if row is None:
                return None  # EOF
            if not row:
                continue

            if self.usecols is None:
                selected = row
            else:
                # assumes row has enough columns; if not, skip
                try:
                    selected = [row[i] for i in self.usecols]
                except IndexError:
                    continue

            try:
                return np.asarray(selected, dtype=self.dtype)
            except ValueError:
                # skip non-numeric lines
                continue

    def next(self) -> np.ndarray:
        """
        Get exactly one sample (row). Auto-rewinds at EOF.
        """
        row = self._read_next_data_row()
        if row is not None:
            return row

        # EOF -> rewind and try again
        self._open_and_position()
        row = self._read_next_data_row()
        if row is None:
            raise ValueError("CSV has no readable data rows after skipping.")
        return row

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Infinite iterator: yields one row at a time forever, auto-rewinding at EOF.
        """
        while True:
            yield self.next()

    def take(self, n: int) -> np.ndarray:
        """Convenience: take n samples and return as (n, d) array."""
        if n <= 0:
            raise ValueError("n must be positive.")
        rows = [self.next() for _ in range(n)]
        return np.vstack(rows)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._reader = None


class csv_posterior_sampler_BikeSharing:
    def __init__(self, csv_dir, num_measures: int = 1, multiplication_factor=1, type: str = "full", usecols: Optional[Union[Sequence[int], range]] = None, skiprows: int = 0):
        if type not in ("full", "split"):
            raise ValueError(f"type must be 'full' or 'split', got '{type}'")
        self.num_measures = num_measures
        self.csv_dir = csv_dir
        self.multiplication_factor = multiplication_factor
        self.type = type
        self.usecols = usecols # G..P if 0-based depends on your earlier choice; this matches your code for bike sharing
        self.skiprows = skiprows

    def set_streamers(self):
        """
        Create streaming samplers (auto pointer) without reservoir sampling.
        - For 'full': returns one streamer.
        - For 'split': returns dict[measure_idx] -> streamer.
        """
        usecols = self.usecols   # G..P if 0-based depends on your earlier choice; this matches your code
        skiprows = self.skiprows

        if self.type == "full":
            csv_filename = os.path.join(self.csv_dir, "posterior_full.csv")
            output_streamer = StreamingCSVSamples(
                csv_filename,
                skiprows=skiprows,
                usecols=usecols,
                has_header=True,
                dtype=float,
            )
            self.output_streamer = output_streamer

        else: 
            output_streamers: Dict[int, StreamingCSVSamples] = {}
            for measure_idx in range(self.num_measures):
                csv_filename = os.path.join(self.csv_dir, f"posterior_split_{measure_idx}.csv")
                output_streamers[measure_idx] = StreamingCSVSamples(
                    csv_filename,
                    skiprows=skiprows,
                    usecols=usecols,
                    has_header=False,
                    dtype=float,
                )
            self.output_streamers = output_streamers

    def sample(self, num_samples: int):
        """
        Backward-friendly: return num_samples samples using streaming (no reservoir).
        """
        if self.type == "full":
            X = self.output_streamer.take(num_samples)
            return self.multiplication_factor * X
        
        else: 
            batch_sample_collection = {}
            for k, streamer in self.output_streamers.items():
                X = streamer.take(num_samples)
                batch_sample_collection[k] = [self.multiplication_factor * row for row in X]
            return batch_sample_collection
        

class csv_input_sampler_SyntheticGeneration:
    def __init__(self, csv_dir, num_measures: int = 1, multiplication_factor=1, usecols: Optional[Union[Sequence[int], range]] = None, skiprows: int = 0):
        self.num_measures = num_measures
        self.csv_dir = csv_dir
        self.multiplication_factor = multiplication_factor
        self.usecols = usecols # G..P if 0-based depends on your earlier choice; this matches your code for bike sharing
        self.skiprows = skiprows

    def set_streamers(self):
        usecols = self.usecols   # G..P if 0-based depends on your earlier choice; this matches your code
        skiprows = self.skiprows

        # split
        output_streamers: Dict[int, StreamingCSVSamples] = {}
        for measure_idx in range(self.num_measures):
            csv_filename = os.path.join(self.csv_dir, f"input_measure_samples_{measure_idx}.csv")
            output_streamers[measure_idx] = StreamingCSVSamples(
                csv_filename,
                skiprows=skiprows,
                usecols=usecols,
                has_header=False,
                dtype=float,
            )
        self.output_streamers = output_streamers

    def sample(self, num_samples: int):
        batch_sample_collection = {}
        for k, streamer in self.output_streamers.items():
            X = streamer.take(num_samples)
            batch_sample_collection[k] = [self.multiplication_factor * row for row in X]
        return batch_sample_collection
    
class csv_auxiliary_sampler_SyntheticGeneration:
    def __init__(self, csv_dir, auxiliary_seed, multiplication_factor=1, usecols: Optional[Union[Sequence[int], range]] = None, skiprows: int = 0):
        self.csv_dir = csv_dir
        self.auxiliary_seed = auxiliary_seed
        self.multiplication_factor = multiplication_factor
        self.usecols = usecols 
        self.skiprows = skiprows

    def set_streamer(self):
        usecols = self.usecols   
        skiprows = self.skiprows

        csv_filename = os.path.join(self.csv_dir, f"auxiliary_measure_seed_{self.auxiliary_seed}.csv")
        output_streamer = StreamingCSVSamples(
            csv_filename,
            skiprows=skiprows,
            usecols=usecols,
            has_header=True,
            dtype=float,
        )
        self.output_streamer = output_streamer

    def sample(self, num_samples: int):
        X = self.output_streamer.take(num_samples)
        return self.multiplication_factor * X
    
    


if __name__ == "__main__":

    # Load Bike_Sharing dataset
    # Example usage
    csv_dir = f"../WB_data/Bike_Sharing"
    num_measures = 5
    multiplication_factor = 1

    # --- full posterior ------------
    sampler = csv_posterior_sampler_BikeSharing(csv_dir, 
                                                num_measures, 
                                                multiplication_factor, 
                                                type="full",
                                                usecols=range(7, 16),
                                                skiprows=52)
    sampler.set_streamers()
    samples = sampler.sample(3) 
    print(samples)
    samples = sampler.sample(3)
    print(samples)

    # --- posterior split ------------
    sampler = csv_posterior_sampler_BikeSharing(csv_dir, 
                                                num_measures, 
                                                multiplication_factor, 
                                                type="split",
                                                usecols=range(7, 16),
                                                skiprows=52)
    sampler.set_streamers()

    samples = sampler.sample(3)
    for measure_idx, measure_samples in samples.items():
        print(f"Measure {measure_idx}:")
        for sample in measure_samples:
            print(sample)

    print("-----")

    samples = sampler.sample(3) 
    for measure_idx, measure_samples in samples.items():
        print(f"Measure {measure_idx}:")
        for sample in measure_samples:
            print(sample)


    # --- Synthetic Generation input measures ------------
    dim = 2
    csv_dir = f"../WB_data/Synthetic_Generation/dim{dim}_data/input_samples/csv_files"
    num_measures = 5

    sampler = csv_input_sampler_SyntheticGeneration(csv_dir, 
                                                   num_measures, 
                                                   multiplication_factor=1,
                                                   usecols=None,
                                                   skiprows=0)
    sampler.set_streamers()
    samples = sampler.sample(3)
    for measure_idx, measure_samples in samples.items():
        print(f"Input Measure {measure_idx}:")
        for sample in measure_samples:
            print(sample)

    print("Done.")

    samples = sampler.sample(3)
    for measure_idx, measure_samples in samples.items():
        print(f"Input Measure {measure_idx}:")
        for sample in measure_samples:
            print(sample)
