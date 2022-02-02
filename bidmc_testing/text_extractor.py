# Extracting data from bidmc text files (Age and Gender)


patient_age = {}
patient_gender_M = {}
patient_gender_F = {}

for x in range(1, 54):
    with open(f'bidmc_{str(x).zfill(2)}_Fix.txt') as pt:
        for rec in pt:
            if rec[0:3] == 'Age':
                patient_age[f'patient{x}'] = rec[5:7]
            if rec[0:3] == 'Gen':
                #patient_gender[f'patient{x}'] = rec[8]
                if rec[8] == 'M':
                    patient_gender_M[f'patient{x}'] = 1
                    patient_gender_F[f'patient{x}'] = 0
                elif rec[8] == 'F':
                    patient_gender_F[f'patient{x}'] = 1
                    patient_gender_M[f'patient{x}'] = 0
                else:
                    patient_gender_F[f'patient{x}'] = 'NaN'
                    patient_gender_M[f'patient{x}'] = 'NaN'


print(patient_age['patient18'])

