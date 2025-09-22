# main.py
from preprocessing import preprocess_pipeline

df_cleaned = preprocess_pipeline(
    input_path="D:\Subjects_Languages\Languages\ML_Internship\Dataset\titanic.csv",
    output_path="D:\Subjects_Languages\Languages\ML_Internship\Output"
)
