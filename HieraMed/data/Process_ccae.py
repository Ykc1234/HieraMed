import os
import pandas as pd
from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets.utils import strptime
from tqdm import tqdm
from pyhealth.medcode import CrossMap


def read_icd_mapping(file_path):
    icd_mapping_df = pd.read_csv(file_path)
    icd9_to_icd10_mapping = {row['icd9cm']: row['icd10cm'] for _, row in icd_mapping_df.iterrows()}
    return icd9_to_icd10_mapping


icd9_to_icd10_mapping = read_icd_mapping('./icd9to10.csv')


# %%
def parse_basic_info(cohort_file):
    patients = dict()

    cohort_df = pd.read_csv(cohort_file, dtype=str)

    for _, row in tqdm(cohort_df.iterrows(), desc="Parsing cohort"):
        patient = Patient(patient_id=row['ENROLID'])
        patients[patient.patient_id] = patient

    return patients


# %%
drug_files = [f"drug{year}.csv" for year in range(12, 13)]


def parse_drug_files(patients, file_list):
    for file_name in file_list:
        df = pd.read_csv(os.path.join(data_dir, file_name), dtype=str)
        for _, row in tqdm(df.iterrows(), desc=f"Parsing {file_name}"):
            visit_id = row['SVCDATE']
            patient_id = row['ENROLID']
            if pd.isna(visit_id) or patient_id not in patients:
                continue
            visit = patients[patient_id].visits.get(visit_id) or Visit(
                visit_id=visit_id,
                patient_id=patient_id
            )
            raw_drug_code = row['NDCNUM']
            drug_code = codemap.map(raw_drug_code)
            if drug_code != []:
                drug_code = drug_code[0]
                event = Event(
                    code=drug_code,
                    table="DRUG",
                    vocabulary="ATC",
                    visit_id=visit_id,
                    patient_id=patient_id
                )
                visit.add_event(event)
                patients[patient_id].visits[visit_id] = visit

    return patients


# patients = parse_drug_files(patients, drug_files)

# %%
import pickle


def save_object(obj, filename):
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


# save_object(patients, 'patients_data.pkl')
def load_object(filename):
    with open(filename, 'rb') as inp:  #
        return pickle.load(inp)


# patients = load_object('patients_data.pkl')

# %%
def parse_inpatient_outpatient_files(patients, file_list, table_name):
    for file_name in file_list:
        df = pd.read_csv(os.path.join(data_dir, file_name), dtype=str)
        for _, row in tqdm(df.iterrows(), desc=f"Parsing {file_name}"):
            visit_id = row['ADMDATE'] if table_name == "INPAT" else row['SVCDATE']
            if pd.isna(visit_id):
                continue
            patient_id = row['ENROLID']
            if patient_id not in patients:
                continue
            visit = patients[patient_id].visits.get(visit_id) or Visit(
                visit_id=visit_id,
                patient_id=patient_id
            )
            patients[patient_id].visits[visit_id] = visit

    return patients


# inpatient_files = [f"inpat{year}.csv" for year in range(12, 13)]
# patients = parse_inpatient_outpatient_files(patients, inpatient_files, "INPAT")

# outpatient_files = [f"outpat{year}.csv" for year in range(12, 13)]
# patients = parse_inpatient_outpatient_files(patients, outpatient_files, "OUTPAT")


# %%
def parse_csv_files(patients, file_list, type, icd_mapping):
    for file_name in file_list:
        df = pd.read_csv(os.path.join(data_dir, file_name), dtype=str)
        for _, row in tqdm(df.iterrows(), desc=f"Parsing {file_name}"):
            patient_id = row['ENROLID']
            if patient_id not in patients:
                continue
            visit_id = row['ADMDATE'] if type == 'inpat' else row['SVCDATE']
            if pd.isna(visit_id):
                continue

            visit = patients[patient_id].visits.get(visit_id) or Visit(
                visit_id=visit_id,
                patient_id=patient_id
            )

            for dx in range(1, (15 if type == 'inpat' else 5)):
                dx_code = row.get(f'DX{dx}')
                if pd.notna(dx_code):
                    dxver = row.get('DXVER', '9')
                    if dxver == '9':
                        dx_code = icd_mapping.get(dx_code, dx_code)
                    visit.add_event(Event(
                        code=dx_code,
                        table=file_name.upper().replace('.CSV', ''),
                        vocabulary='ICD10',
                        visit_id=visit_id,
                        patient_id=patient_id
                    ))

            for pr in range(1, (16 if type == 'inpat' else 2)):
                pr_code = row.get(f'PROC{pr}')
                if pd.notna(pr_code):
                    pr_code = icd_mapping.get(pr_code, pr_code)
                    visit.add_event(Event(
                        code=pr_code,
                        table=file_name.upper().replace('.CSV', ''),
                        vocabulary='ICD10',
                        visit_id=visit_id,
                        patient_id=patient_id
                    ))

            patients[patient_id].visits[visit_id] = visit

    return patients


# inpatient_files = [f"inpat{year}.csv" for year in range(12, 13)]
# patients = parse_csv_files(patients, inpatient_files, 'inpat', icd9_to_icd10_mapping)

# outpatient_files = [f"outpat{year}.csv" for year in range(12, 13)]
# patients = parse_csv_files(patients, outpatient_files, 'outpat', icd9_to_icd10_mapping)


# %%
from pyhealth.datasets import SampleEHRDataset as SampleEHRDataset
# %%
import pickle
from joblib import dump, load


class CustomDataset():
    def __init__(self, root, dev, processed_file='processed_copd.pkl'):
        # super().__init__(root)
        self.root = root
        end_year = 15 if dev else 17
        self.processed_file = os.path.join(root, processed_file)
        self.drug_files = [f"drug{year}.csv" for year in range(12, end_year)]
        self.inpatient_files = [f"inpat{year}.csv" for year in range(12, end_year)]
        self.outpatient_files = [f"outpat{year}.csv" for year in range(12, end_year)]
        self.dataset_name = 'COPD'

    def load_or_parse_tables(self):
        # Check if processed data file exists
        if os.path.isfile(self.processed_file):
            # Load the processed data
            with open(self.processed_file, 'rb') as file:
                # self.patients = pickle.load(file)
                self.patients = load(self.processed_file)

        else:
            # Process and parse the data
            patients = self.parse_tables()
            self.patients = patients

            dump(self.patients, self.processed_file)

    def parse_tables(self):
        patients = dict()
        icd9_to_icd10_mapping = read_icd_mapping('./icd9to10.csv')
        # Add custom parse functions here
        patients = self.parse_cohort(patients)
        patients = self.parse_drug_files(patients, self.drug_files)
        patients = self.parse_inpatient_outpatient_files(patients, self.inpatient_files, "INPAT")
        patients = self.parse_inpatient_outpatient_files(patients, self.outpatient_files, "OUTPAT")
        patients = self.parse_csv_files(patients, self.inpatient_files, 'inpat', icd9_to_icd10_mapping)
        patients = self.parse_csv_files(patients, self.outpatient_files, 'outpat', icd9_to_icd10_mapping)

        return patients

    def parse_cohort(self, patients):
        cohort_df = pd.read_csv(os.path.join(self.root, "Cohort.csv"), dtype=str)
        for _, row in cohort_df.iterrows():
            patient = Patient(patient_id=row['ENROLID'])
            patients[patient.patient_id] = patient
        return patients

    def parse_drug_files(self, patients, file_list):
        for file_name in file_list:
            df = pd.read_csv(os.path.join(data_dir, file_name), dtype=str)
            for _, row in tqdm(df.iterrows(), desc=f"Parsing {file_name}"):
                visit_id = row['SVCDATE']
                patient_id = row['ENROLID']
                if pd.isna(visit_id) or patient_id not in patients:
                    continue
                visit = patients[patient_id].visits.get(visit_id) or Visit(
                    visit_id=visit_id,
                    patient_id=patient_id
                )
                raw_drug_code = row['NDCNUM']
                drug_code = codemap.map(raw_drug_code)
                if drug_code != []:
                    drug_code = drug_code[0]
                    event = Event(
                        code=drug_code,
                        table="prescriptions",
                        vocabulary="ATC",
                        visit_id=visit_id,
                        patient_id=patient_id
                    )
                    visit.add_event(event)
                    patients[patient_id].visits[visit_id] = visit

        return patients

    def parse_inpatient_outpatient_files(self, patients, file_list, table_name):
        for file_name in file_list:
            df = pd.read_csv(os.path.join(data_dir, file_name), dtype=str)
            for _, row in tqdm(df.iterrows(), desc=f"Parsing {file_name}"):
                visit_id = row['ADMDATE'] if table_name == "INPAT" else row['SVCDATE']
                if pd.isna(visit_id):
                    continue
                patient_id = row['ENROLID']
                if patient_id not in patients:
                    continue

                visit = patients[patient_id].visits.get(visit_id) or Visit(
                    visit_id=visit_id,
                    patient_id=patient_id
                )
                patients[patient_id].visits[visit_id] = visit

        return patients

    def parse_csv_files(self, patients, file_list, type, icd_mapping):
        for file_name in file_list:
            df = pd.read_csv(os.path.join(data_dir, file_name), dtype=str)
            for _, row in tqdm(df.iterrows(), desc=f"Parsing {file_name}"):
                patient_id = row['ENROLID']
                if patient_id not in patients:
                    continue

                visit_id = row['ADMDATE'] if type == 'inpat' else row['SVCDATE']
                if pd.isna(visit_id):
                    continue

                visit = patients[patient_id].visits.get(visit_id) or Visit(
                    visit_id=visit_id,
                    patient_id=patient_id
                )

                for dx in range(1, (15 if type == 'inpat' else 5)):
                    dx_code = row.get(f'DX{dx}')
                    if pd.notna(dx_code):
                        dxver = row.get('DXVER', '9')
                        if dxver == '9':
                            dx_code = icd_mapping.get(dx_code, dx_code)
                        visit.add_event(Event(
                            code=dx_code,
                            table='diagnoses_icd',
                            vocabulary='ICD10',
                            visit_id=visit_id,
                            patient_id=patient_id
                        ))

                for pr in range(1, (16 if type == 'inpat' else 2)):
                    pr_code = row.get(f'PROC{pr}')
                    if pd.notna(pr_code):
                        pr_code = icd_mapping.get(pr_code, pr_code)
                        visit.add_event(Event(
                            code=pr_code,
                            table='procedures_icd',
                            vocabulary='ICD10',
                            visit_id=visit_id,
                            patient_id=patient_id
                        ))

                patients[patient_id].visits[visit_id] = visit

        return patients

    def set_task(
            self,
            task_fn,
            task_name
    ):
        if task_name is None:
            task_name = task_fn.__name__
        samples = []
        for patient_id, patient in tqdm(
                self.patients.items(), desc=f"Generating samples for {task_name}"
        ):
            samples.extend(task_fn(patient))

        # Save the samples to the class instance for further use
        self.samples = samples

        sample_dataset = SampleEHRDataset(
            samples=samples,
            dataset_name=self.dataset_name,
            task_name=task_name,
        )
        return sample_dataset


# %%
data_dir = ""
codemap = CrossMap.load("NDC", "ATC")
custom_dataset = CustomDataset(root=data_dir, dev=True)
custom_dataset.load_or_parse_tables()