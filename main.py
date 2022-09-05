from smile_preprocessing import smile_data_clean
from model import create_model

smile_df = smile_data_clean()
create_model(smile_df)