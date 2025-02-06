class RegionAttributes:
    def __init__(self, language: str, dosage: str, dignostic: str, symptoms: str, medicine_name: str, text: str, personal_info: str, numeric_data: str):
        self.language = language
        self.dosage = dosage
        self.dignostic = dignostic
        self.symptoms = symptoms
        self.medicine_name = medicine_name
        self.text = text
        self.personal_info = personal_info
        self.numeric_data = numeric_data

    def to_dict(self):
        return {
            "Language": self.language,
            "Dosage": self.dosage,
            "Dignostic": self.dignostic,
            "Symptoms": self.symptoms,
            "Medicine Name": self.medicine_name,
            "Text": self.text,
            "Personal Information": self.personal_info,
            "Numeric Data": self.numeric_data,
        }

    def __repr__(self):
        return f"RegionAttributes(Language={self.language}, Dosage={self.dosage}, Dignostic={self.dignostic}, Symptoms={self.symptoms}, Medicine Name={self.medicine_name}, Text={self.text}, Personal Info={self.personal_info}, Numeric Data={self.numeric_data})"
