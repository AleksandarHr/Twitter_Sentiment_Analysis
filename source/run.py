from helpers import *
import os
import pandas as pd


def prepare_submission(model, test_data, save_path, batch_size=100):
    """
    creates
    @param model: pre-trained model used for prediction
    @param test_data: pandas data frame containing the test data
    @param save_path: path indicating where to save the submission file
    @param batch_size: batch size
    @return: None
    """
    test_tweet = test_data['tweet']
    predictions = []

    # Iterate over test data
    for i in range(int(test_tweet.shape[0] / batch_size)):
        batch_text = test_tweet.iloc[i*batch_size:(i+1)*batch_size]
        batch_prediction = model.forward(batch_text)
        predictions.append(batch_prediction)

    # Make prediction conform to submission
    predictions = torch.cat(predictions)
    predictions = torch.argmax(predictions, dim=1)
    predictions[predictions == 0] = -1
    predictions = predictions.cpu()
    submission_df = pd.DataFrame({'Id': np.arange(1, predictions.shape[0] + 1),
                            'Prediction': predictions})
    submission_df.to_csv(os.path.join(save_path, 'submission.csv'), index=False)


def main(data_path, model_path):
    # Load online test data
    test_online = pd.read_csv(os.path.join(data_path, 'test_online.txt'))

    # Select available device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model and ensure compatibility
    model = torch.load(model_path, map_location=device)
    model.device = device
    model.is_lstm = False

    # Make prediction
    prepare_submission(model, test_online, data_path)


if __name__ == '__main__':
    data_path = '../Data/'
    model_path = '../models/GRUNet.bin'

    main(data_path, model_path)
