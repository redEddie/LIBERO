import copy

import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from PIL import Image
from robomimic.utils.dataset import SequenceDataset
from torch.utils.data import Dataset

"""
    Helper function from Robomimic to read hdf5 demonstrations into sequence dataset

    ISSUE: robomimic's SequenceDataset has two properties: seq_len and frame_stack,
    we should in principle use seq_len, but the paddings of the two are different.
    So that's why we currently use frame_stack instead of seq_len.
"""


def get_dataset(
    dataset_path,
    obs_modality,
    initialize_obs_utils=True,
    seq_len=1,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",  # "low_dim": 저차원 데이터(joint, gripper)만 캐싱, 이미지는 매 step 디스크에서 읽음
                                #            → RAM 절약. 단 h5py 핸들이 열려있어 num_workers=0 필수.
                                # "all"    : raw HDF5 + 모든 getitem 결과까지 이중 캐싱.
                                #            image-based 데이터셋은 sliding-window 중복으로 수백 GB 필요 → 사용 금지.
                                # "none"   : 캐싱 없음 (RAM 최소, 디스크 I/O 최대)
    *args,
    **kwargs
):

    if initialize_obs_utils:
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})

    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )

    seq_len = seq_len
    filter_key = filter_key
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=shape_meta["all_obs_keys"],
        dataset_keys=["actions"],
        load_next_obs=False,
        frame_stack=frame_stack,
        seq_length=seq_len,  # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=False,
        hdf5_normalize_obs=None,
        filter_by_attribute=filter_key,  # can optionally provide a filter key here
    )
    return dataset, shape_meta


class SequenceVLDataset(Dataset):
    def __init__(self, sequence_dataset, task_emb):
        self.sequence_dataset = sequence_dataset
        self.task_emb = task_emb
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return_dict = self.sequence_dataset.__getitem__(idx)
        return_dict["task_emb"] = self.task_emb
        return return_dict


class GroupedTaskDataset(Dataset):
    def __init__(self, sequence_datasets, task_embs):
        self.sequence_datasets = sequence_datasets
        self.task_embs = task_embs
        self.group_size = len(sequence_datasets)
        self.n_demos = sum([x.n_demos for x in self.sequence_datasets])
        self.total_num_sequences = sum(
            [x.total_num_sequences for x in self.sequence_datasets]
        )
        self.lengths = [len(x) for x in self.sequence_datasets]
        self.task_group_size = len(self.sequence_datasets)

        # create a map that maps the current idx of dataloader to original task data idx
        # imagine we have task 1,2,3, with sizes 3,5,4, then the idx looks like
        # task-1  task-2  task-3
        #   0       1       2
        #   3       4       5
        #   6       7       8
        #           9       10
        #           11
        # by doing so, when we concat the dataset, every task will have equal number of demos
        self.map_dict = {}
        sizes = np.array(self.lengths)
        row = 0
        col = 0
        for i in range(sum(sizes)):
            while sizes[col] == 0:
                col = col + 1
                if col >= self.task_group_size:
                    col -= self.task_group_size
                    row += 1
            self.map_dict[i] = (row, col)
            sizes[col] -= 1
            col += 1
            if col >= self.task_group_size:
                col -= self.task_group_size
                row += 1
        self.n_total = sum(self.lengths)

    def __len__(self):
        return self.n_total

    def __get_original_task_idx(self, idx):
        return self.map_dict[idx]

    def __getitem__(self, idx):
        oi, oti = self.__get_original_task_idx(idx)
        return_dict = self.sequence_datasets[oti].__getitem__(oi)
        return_dict["task_emb"] = self.task_embs[oti]
        return return_dict


class H5pyPicklableVLDataset(Dataset):
    """
    SequenceVLDataset wrapper that enables num_workers > 0 without extra RAM.

    문제: robomimic의 SequenceDataset은 내부에 열린 h5py 파일 핸들을 보유.
         spawn 방식 멀티프로세싱에서 DataLoader 워커로 dataset을 전달할 때
         pickle이 필요한데, h5py 핸들은 pickle 불가 → TypeError 발생.

    해결: __getstate__에서 h5py 핸들을 닫고(None), __setstate__ 후
         robomimic의 lazy property가 워커 프로세스에서 자동으로 재오픈.

    메모리: hdf5_cache_mode="low_dim" 기준, 이미지는 각 워커에서 디스크에서 읽으므로
            RAM 추가 사용 없음. num_workers 수만큼 병렬 I/O 가능.
    """

    def __init__(self, vl_dataset):
        self._vl = vl_dataset
        self.task_emb = vl_dataset.task_emb
        self.n_demos = vl_dataset.n_demos
        self.total_num_sequences = vl_dataset.total_num_sequences

    def __len__(self):
        return len(self._vl)

    def __getstate__(self):
        state = self.__dict__.copy()
        # pickle 전 h5py 핸들 닫기 → _hdf5_file = None
        # (robomimic의 hdf5_file property가 워커에서 자동 재오픈)
        state["_vl"].sequence_dataset.close_and_delete_hdf5_handle()
        return state

    def __setstate__(self, state):
        # 복원 후 h5py는 첫 __getitem__ 호출 시 lazy하게 재오픈됨
        self.__dict__.update(state)

    def __getitem__(self, idx):
        return self._vl[idx]


class TruncatedSequenceDataset(Dataset):
    def __init__(self, sequence_dataset, buffer_size):
        self.sequence_dataset = sequence_dataset
        self.buffer_size = buffer_size

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, idx):
        return self.sequence_dataset.__getitem__(idx)
