from PredictiveOutlierExplanationBenchmark.src.models.OutlierDetector import Detector


def detect_outliers(DetectorConf, dataset):
    detector = Detector(DetectorConf)
    detector.train(dataset.get_data())
    scores = detector.score_samples()
    print(scores)
