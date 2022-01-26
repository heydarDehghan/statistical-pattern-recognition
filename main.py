from services import *
from classification import Bayesian, Quadratic, NaiveBayes

if __name__ == '__main__':
    # region partOne

    # for FILE_NUMBER in range(2):
        # path_list = [('dataset/BC-Train1.csv', 'dataset/BC-Test1.csv'),
        #              ('dataset/BC-Train2.csv', 'dataset/BC-Test2.csv')]
        #
        # data_train = load_data(path_list[FILE_NUMBER][0])
        # data_test = load_data(path_list[FILE_NUMBER][1])
        #
        # data = Data
        #
        # data.trainData = data_train[:, 0:-1]
        # data.trainLabel = data_train[:, -1]
        #
        # data.testData = data_test[:, 0:-1]
        # data.testLabel = data_test[:, -1]
        #
        # print(f'\ndataset with file {FILE_NUMBER} created ')
        #
        # model = Bayesian(data)
        # model.fit()
        # train_predict_label = model.predict(data.trainData)
        # test_predict_label = model.predict(data.testData)
        #
        # print(f"Train Accuracy in file {FILE_NUMBER} : {accuracy_metric(data.trainLabel, train_predict_label)} ")
        # print(f"test Accuracy in file {FILE_NUMBER} : {accuracy_metric(data.testLabel, test_predict_label)}  ")
        #
        # test_precision, test_recall, test_fscore = calculate_metrics(train_predict_label, data.trainLabel)
        #
        # print(f"test precision in file {FILE_NUMBER} : {test_precision} ")
        # print(f"test recall in file {FILE_NUMBER} : {test_recall} ")
        # print(f"test fscore in file {FILE_NUMBER} : {test_fscore} \n\n ")
        #
        # # Plots
        # plot_dec_boundary(data.trainData, data.trainData, train_predict_label, data.trainLabel, model.means,
        #                   model.covariance_list,
        #                   model.priors, f"train data file {FILE_NUMBER}")
        #
        # print(1)
        #
        # plot_dec_boundary(data.testData, data.testData, test_predict_label, data.testLabel, model.means,
        #                   model.covariance_list,
        #                   model.priors, f"test data file {FILE_NUMBER}")
        #
        # print(1)
        # plot_pdfs(data.trainData, model.means, model.covariance_list, model.class_name_list, model.priors,
        #           data.trainLabel)
        # plt.show()
        #
        # plot_pdfs(data.testData, model.means, model.covariance_list, model.class_name_list, model.priors,
        #           data.testLabel)
        # plt.show()

    # endregion partOne

    # region Quadratic

    listdata = ['dataset/multiClass/dataset1.csv', 'dataset/multiClass/dataset2.csv']

    for index, path in enumerate(listdata):
        rawData = load_data(path)
        data = Data(rawData, bias=False)

        model_q = Quadratic(data)
        model_q.fix()

        predicted_train = model_q.predict(data.trainData)
        predicted_test = model_q.predict(data.testData)

        print(f"Dataset {index}, Train ACC = {accuracy_metric(data.trainLabel, predicted_train)}")
        print(f"Dataset {index}, test ACC = {accuracy_metric(data.testLabel, predicted_test)} \n")

        results = decision_boundaries1(data.trainData, predicted_train, model_q.means, model_q.covariance_matrix, model_q.priors)
        plot_dec_boundary1(data.trainData, results, predicted_train, data.trainLabel, model_q.means, model_q.covariance_matrix, model_q.priors, "train data")
        plot_pdfs1(data.trainData, results, predicted_train, model_q.means, model_q.covariance_matrix, model_q.class_name_list, model_q.priors, data.trainData)


    # endregion Quadratic


    list_path = ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']

    for path in list_path:
        raw_data = load_word_data(f'dataset/Sentiment Labelled Sentences/{path}')
        data = Data
        data.trainData, data.testData, data.trainLabel, data.testLabel = train_test_split(raw_data['line'],
                                                                                          raw_data['label'],
                                                                                          test_size=0.2,
                                                                                          stratify=raw_data['label'],
                                                                                          random_state=42)

        model_niv = NaiveBayes(data)
        predict_train = pd.DataFrame(model_niv.fit())

        print(f'dataset name is  ==> {path}')
        print(f'Train Accuracy')
        print(f'total : {accuracy_metric(np.array(data.trainLabel), np.array(predict_train))}')
        y_train_0 = data.trainLabel[data.trainLabel.isin([0])]
        y_train_1 = data.trainLabel[data.trainLabel.isin([1])]
        predicted_0 = predict_train[predict_train.isin([0])]
        predicted_1 = predict_train[predict_train.isin([1])]
        print(f"Class 0 : {accuracy_metric(np.array(y_train_0), np.array(predicted_0))}")
        print(f"Class 1 : {accuracy_metric(np.array(y_train_1), np.array(predicted_1))} \n")

        predict_test = pd.DataFrame(model_niv.predict(data.testData))
        print(f'Test Accuracy')
        print(f'total : {accuracy_metric(np.array(data.testLabel), np.array(predict_test))}')
        y_test_0 = data.testLabel[data.testLabel.isin([0])]
        y_test_1 = data.testLabel[data.testLabel.isin([1])]
        predicted_test_0 = predict_test[predict_test.isin([0])]
        predicted_test_1 = predict_test[predict_test.isin([1])]
        print(f"Class 0 : {accuracy_metric(np.array(y_test_0), np.array(predicted_test_0))}")
        print(f"Class 1 : {accuracy_metric(np.array(y_test_1), np.array(predicted_test_1))} \n\n")
