
class ResultsHolder:

    def __init__(self):
        self.metrics_dict = {}

    def update(self, fsel, classifier, metric_id):
        conf_id = fsel.get_id() + '_' + classifier.get_id()


