import os
import pandas as pd

class Preprocessing:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.data = pd.read_csv(self.data_dir+'train_ratings.csv')
        self.unique_sid = None
        self.unique_uid = None
        self.show2id = None
        self.profile2id = None

        self.output_dir = output_dir

        # Run the preprocessing pipeline during initialization
        self.run()


    def prepare_mappings(self):
        self.unique_sid = pd.unique(self.data['item'])
        self.unique_uid = pd.unique(self.data['user'])
        self.show2id = {sid: i for i, sid in enumerate(self.unique_sid)}
        self.profile2id = {pid: i for i, pid in enumerate(self.unique_uid)}

    def save_mappings(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        with open(os.path.join(self.output_dir, 'unique_sid.txt'), 'w') as f:
            for sid in self.unique_sid:
                f.write(f'{sid}\n')

        with open(os.path.join(self.output_dir, 'unique_uid.txt'), 'w') as f:
            for uid in self.unique_uid:
                f.write(f'{uid}\n')

    def numerize(self, data):
        uid = data['user'].map(self.profile2id)
        sid = data['item'].map(self.show2id)
        return pd.DataFrame({'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

    def save_numerized_data(self):
        train_data = self.numerize(self.data)
        train_data.to_csv(os.path.join(self.output_dir, 'train.csv'), index=False)

    def run(self):
        """Run the entire preprocessing pipeline."""

        print("Preparing mappings...")
        self.prepare_mappings()
        print("Saving mappings...")
        self.save_mappings()
        print("Numerizing and saving data...")
        self.save_numerized_data()
        print("Preprocessing complete.")