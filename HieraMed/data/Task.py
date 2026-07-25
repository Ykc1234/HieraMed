from pyhealth.data import  Patient
from torch.utils.data import  Dataset
from data.GraphConstruction import *
from tqdm import *
from pyhealth.medcode import CrossMap
mapping = CrossMap("ICD10CM", "CCSCM")
mapping3 = CrossMap("ICD9CM", "CCSCM")
def diag_prediction_mimic4_fn(patient: Patient):
    samples = []
    visit_ls = list(patient.visits.keys())
    for i in range(len(visit_ls)):
        visit = patient.visits[visit_ls[i]]
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        cond_ccs = []
        for con in conditions:
            if mapping.map(con):
                cond_ccs.append(mapping.map(con)[0]) 

        if len(cond_ccs) * len(procedures) * len(drugs) == 0:
            continue
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conds_hist": cond_ccs,
                "procedures": procedures,
                "adm_time" : visit.encounter_time.strftime("%Y-%m-%d %H:%M"),
                "drugs": drugs,
                "conditions": cond_ccs,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conds_hist"] = [samples[0]["conds_hist"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]
    samples[0]["adm_time"] = [samples[0]["adm_time"]]

    for i in range(1, len(samples)):
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [
            samples[i]["drugs"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["conds_hist"] = samples[i - 1]["conds_hist"] + [
            samples[i]["conds_hist"]
        ]
        samples[i]["adm_time"] = samples[i - 1]["adm_time"] + [
            samples[i]["adm_time"]
        ]

    for i in range(len(samples)):
        samples[i]["conds_hist"][i] = []

    return samples[1:]


def diag_prediction_mimic3_fn(patient: Patient):
    samples = []
    visit_ls = list(patient.visits.keys())
    for i in range(len(visit_ls)):
        visit = patient.visits[visit_ls[i]]
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        cond_ccs = []
        for con in conditions:
            if mapping3.map(con):
                cond_ccs.append(mapping3.map(con)[0]) 
        if len(cond_ccs) * len(procedures) * len(drugs) == 0:
            continue
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conds_hist": cond_ccs,
                "procedures": procedures,
                "adm_time" : visit.encounter_time.strftime("%Y-%m-%d %H:%M"),
                "drugs": drugs,
                "conditions": cond_ccs,
            }
        )
    if len(samples) < 2:
        return []
    # add history
    samples[0]["drugs"] = [samples[0]["drugs"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["conds_hist"] = [samples[0]["conds_hist"]]
    samples[0]["adm_time"] = [samples[0]["adm_time"]]

    for i in range(1, len(samples)):
        samples[i]["conds_hist"] = samples[i - 1]["conds_hist"] + [
            samples[i]["conds_hist"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [
            samples[i]["drugs"]
        ]
        samples[i]["adm_time"] = samples[i - 1]["adm_time"] + [
            samples[i]["adm_time"]
        ]
    for i in range(len(samples)):
        samples[i]["conds_hist"][i] = []
    return samples[1:]

def mortality_prediction_mimic4_fn(patient: Patient):
    samples = []
    visit_ls = list(patient.visits.keys())
    for i in range(len(visit_ls)):
        visit = patient.visits[visit_ls[i]]
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        cond_ccs = []
        for con in conditions:
            if mapping.map(con):
                cond_ccs.append(mapping.map(con)[0])

        if len(cond_ccs) * len(procedures) * len(drugs) == 0:
            continue

        if visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(visit.discharge_status)
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": cond_ccs,
                "procedures": procedures,
                "adm_time" : visit.encounter_time.strftime("%Y-%m-%d %H:%M"),
                "drugs": drugs,
                "label": [mortality_label],
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]
    samples[0]["adm_time"] = [samples[0]["adm_time"]]

    for i in range(1, len(samples)):
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [
            samples[i]["drugs"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["adm_time"] = samples[i - 1]["adm_time"] + [
            samples[i]["adm_time"]
        ]

    samples = [samples[-1]]

    return samples

def mortality_prediction_mimic3_fn(patient: Patient):
    samples = []
    visit_ls = list(patient.visits.keys())
    for i in range(len(visit_ls)):
        visit = patient.visits[visit_ls[i]]
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        cond_ccs = []
        for con in conditions:
            if mapping3.map(con):
                cond_ccs.append(mapping3.map(con)[0])
        if len(cond_ccs) * len(procedures) * len(drugs) == 0:
            continue
        if visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(visit.discharge_status)
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": cond_ccs,
                "procedures": procedures,
                "adm_time": visit.encounter_time.strftime("%Y-%m-%d %H:%M"),
                "drugs": drugs,
                "label": [mortality_label],
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]
    samples[0]["adm_time"] = [samples[0]["adm_time"]]

    for i in range(1, len(samples)):
        samples[i]["drugs"] = samples[i - 1]["drugs"] + [
            samples[i]["drugs"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["adm_time"] = samples[i - 1]["adm_time"] + [
            samples[i]["adm_time"]
        ]
    samples = [samples[-1]]

    return samples

class MMSDataset(Dataset):
    def __init__(self, dataset, tokenizer, dim, device, trans_dim=0, di=False):
        self.sequence_dataset = dataset.samples
        self.tokenizer = tokenizer
        self.trans_dim = trans_dim
        self.di = di
        self.dim = dim
        self.device = device
        self.graph_data = PatientGraphSequeces(self.tokenizer, self.sequence_dataset, dim=self.dim, device = self.device, trans_dim=self.trans_dim, di=self.di).all_data

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        sequence_data = self.sequence_dataset[idx]
        graph_data = self.graph_data[idx]
        return sequence_data, graph_data

class MMDataset(Dataset):
    def __init__(self, dataset, tokenizer, dim, device, trans_dim=0, di=False):
        self.sequence_dataset = dataset.samples
        self.tokenizer = tokenizer
        self.trans_dim = trans_dim
        self.di = di
        self.dim = dim
        self.device = device
        self.graph_data = PatientGraph(self.tokenizer, self.sequence_dataset, dim=self.dim, device = self.device, trans_dim=self.trans_dim, di=self.di).all_data

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        sequence_data = self.sequence_dataset[idx]
        graph_data = self.graph_data[idx]
        return sequence_data, graph_data