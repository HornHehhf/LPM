import torch
import numpy as np
from scipy import spatial
import sys
from torch.autograd import Variable

from utils import set_random_seed, get_minibatches_idx
from models import ResNet18, VGG
from data import load_data_from_pickle
from train_models import simple_test_batch

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def load_model(config):
    if config['model'] == 'ResNet18':
        model = ResNet18(color_channel=config['color_channel'])
    elif config['model'] == 'VGG11':
        model = VGG('VGG11', color_channel=config['color_channel'])
    elif config['model'] == 'VGG13':
        model = VGG('VGG13', color_channel=config['color_channel'])
    else:
        print('wrong model option')
        model = None
    model_path = config['dir_path'] + '/models/' + config['data'] + '_' + config['model'] + '_t1=' + \
                 str(config['t1']) + '_R=' + config['R'] + "_" + config['fixed'] + '.pt'
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    return model


def get_features(trainloader, model, config):
    total_features = []
    total_labels = []
    minibatches_idx = get_minibatches_idx(len(trainloader), minibatch_size=config['simple_test_batch_size'],
                                          shuffle=False)
    for minibatch in minibatches_idx:
        inputs = torch.Tensor(np.array([list(trainloader[x][0].cpu().numpy()) for x in minibatch]))
        targets = torch.Tensor(np.array([list(trainloader[x][1].cpu().numpy()) for x in minibatch]))
        inputs, targets = Variable(inputs.cuda()).squeeze(1), Variable(targets.cuda()).squeeze()
        features = model.get_features(inputs)
        total_features.extend(features.cpu().data.numpy().tolist())
        total_labels.extend(targets.cpu().data.numpy().tolist())
    total_features = np.array(total_features)
    total_labels = np.array(total_labels)
    print('total features', total_features.shape)
    print('total labels', total_labels.shape)
    avg_feature = np.mean(total_features, axis=0)
    # print('avg feature', np.linalg.norm(avg_feature))
    centralized_features = total_features - avg_feature
    feature_norm = np.square(np.linalg.norm(centralized_features, axis=1))
    class_features = []
    feature_norm_list = []
    for i in range(10):
        mask_index = (total_labels == i)
        mask_index = mask_index.reshape(len(mask_index), 1)
        # print('mask index', mask_index)
        if config['R'] == 'inf' and i == config['t1']:
            break
        class_features.append(np.sum(total_features * mask_index, axis=0) / np.sum(mask_index.reshape(-1)))
        feature_norm_list.append(np.sum(feature_norm * mask_index.reshape(-1)) / np.sum(mask_index.reshape(-1)))

    class_features = np.array(class_features)
    # print('original class features', class_features)
    class_features = np.array(class_features) - avg_feature
    # print('centralized class features', class_features)
    print('feature norm list', feature_norm_list)
    print('avg square feature norm', np.mean(feature_norm_list))
    return class_features


def analyze_collapse(linear_weights, config, option='weights'):
    num_classes = len(linear_weights)
    weight_norm = [np.linalg.norm(linear_weights[i]) for i in range(num_classes)]
    cos_matrix = np.zeros((num_classes, num_classes))
    between_class_cos = []
    for i in range(num_classes):
        for j in range(num_classes):
            cos_value = 1 - spatial.distance.cosine(linear_weights[i], linear_weights[j])
            cos_matrix[i, j] = cos_value
            if i != j:
                between_class_cos.append(cos_value)
    weight_norm = np.array(weight_norm)
    print('{0} avg square norm'.format(option), np.mean(np.square(weight_norm)))
    between_class_cos = np.array(between_class_cos)
    print('{0} norm'.format(option), weight_norm)
    print('cos {0} matrix'.format(option), cos_matrix)
    print('between class {0} cosine'.format(option), between_class_cos)
    print('std {0} norm over avg {0} norm'.format(option), np.std(weight_norm) / np.mean(weight_norm))
    print('avg between-class {0} cosine'.format(option), np.mean(between_class_cos))
    print('std between-class {0} cosine'.format(option), np.std(between_class_cos))
    print('avg {0} cosine to -1/(C-1)'.format(option), np.mean(np.abs(between_class_cos + 1 / (num_classes - 1))))
    # compute between-class cosine for small classes
    if config['t1'] != len(linear_weights):
        t1 = config['t1']
        print('{0} cosine for small classes'.format(option), cos_matrix[t1:, t1:])
        between_class_cos_small = []
        for i in range(10)[t1:]:
            for j in range(10)[t1:]:
                if i != j:
                    between_class_cos_small.append(cos_matrix[i, j])
        print('between-calss {0} cosine for small classes'.format(option), between_class_cos_small)
        print('avg between-class {0} cosine for small classes'.format(option), np.mean(between_class_cos_small))
        print('std between-class {0} cosine for small classes'.format(option), np.std(between_class_cos_small))
        print('std {0} norm over avg {0} norm for small classes'.format(option), np.std(weight_norm[t1:]) /
              np.mean(weight_norm[t1:]))


def analyze_dual(linear_weights, class_features):
    n_class = len(class_features)
    linear_weights = linear_weights[:n_class]
    linear_weights = linear_weights / np.linalg.norm(linear_weights)
    class_features = class_features / np.linalg.norm(class_features)
    # print('normalized linear weights', linear_weights)
    # print('normalized class features', class_features)
    print('dual distance', np.linalg.norm(linear_weights - class_features))
    print('dual distance square', np.square(np.linalg.norm(linear_weights - class_features)))


if __name__ == '__main__':
    data_option = sys.argv[1].split('=')[1]
    model_option = sys.argv[2].split('=')[1]
    t1 = int(sys.argv[3].split('=')[1])
    R = (sys.argv[4].split('=')[1])
    config = {'dir_path': '/path/to/working/dir', 'data': data_option, 'model': model_option, 't1': t1, 'R': R,
              'simple_test_batch_size': 100, 'fixed': 'big', 'weight_decay': 5e-4}
    # fixed: big/small
    if data_option == 'fashion_mnist':
        config['color_channel'] = 1
    else:
        config['color_channel'] = 3
    set_random_seed(666)
    print('load data from pickle')
    train_data, test_data = load_data_from_pickle(config)
    print('load model')
    model = load_model(config)
    test_res, test_big, test_small, test_confusion_matrix = simple_test_batch(test_data, model, config)
    print('test accuracy', test_res, test_big, test_small)
    print('test confusion matrix\n', test_confusion_matrix)
    linear_weights = model.classifier.weight.cpu().data.numpy()
    print('analyze weights', linear_weights.shape)
    analyze_collapse(linear_weights, config, option='weights')
    class_features = get_features(train_data, model, config)
    print('analyze features')
    analyze_collapse(class_features, config, option='features')
    print('analyze the duality of weights and features')
    analyze_dual(linear_weights, class_features)



