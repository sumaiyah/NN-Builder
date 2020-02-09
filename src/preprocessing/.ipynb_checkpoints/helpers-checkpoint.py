import pandas as pd

def save_data(X, y, col_names, filename):
    # Save data to file
    data = pd.DataFrame(data=X, columns=col_names[0:len(col_names) - 1])
    data['target'] = y
    data.to_csv(filename, index=False)