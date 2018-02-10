import scipy.io
import numpy as np
import pickle

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)

def main():
    train_data = scipy.io.loadmat(r'F:\svhn\extra_32x32')
    test_data = scipy.io.loadmat(r'F:\svhn\test_32x32')
    train_x = np.transpose(train_data['X'], [3, 0, 1, 2])
    test_x = np.transpose(test_data['X'], [3, 0, 1, 2])
    train_y = train_data['y'].reshape(-1)
    train_y[np.where(train_y == 10)] = 0
    test_y = test_data['y'].reshape(-1)
    test_y[np.where(test_y == 10)] = 0
    train = {'X': train_x,
             'y': train_y}
    test = {'X': test_x,
            'y': test_y}
        
    save_pickle(train, r'F:\svhn\train.pkl')
    save_pickle(test, r'F:\svhn\test.pkl')
    
    
if __name__ == "__main__":
    main()