###############################################################################
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this program. If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
###############################################################################

from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
import math
from sklearn.model_selection import train_test_split


from sciutil import SciUtil
from scivae.util import convert_str_labels_to_ints


class Validate(object):

    def __init__(self, vae, class_labels: list, train_percentage=85.0, random_state=17, sciutil=None):
        self.u = sciutil if sciutil is not None else SciUtil()
        self.vae = vae
        self.class_labels = convert_str_labels_to_ints(class_labels)
        self.train_split = train_percentage/100.0
        self.data = vae.get_encoded_data()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        # Generate the training and testing data
        self.random_state = random_state
        self.generate_test_train()

    def generate_test_train(self):
        train_size = int(math.ceil(len(self.data) * self.train_split)) - 1
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.class_labels,
                                                                                random_state=self.random_state,
                                                                                train_size=train_size)

    def predict(self, method: str, metric: str, arg=None) -> float:
        classifier = None
        if method == 'svm':
            classifier = self.svm()
        elif method == 'rf':
            classifier = self.rf(arg)
        elif method == 'lin_reg':
            classifier = self.lin_reg()
            classifier.fit(self.X_train, self.y_train)
            return classifier.score(self.X_test, self.y_test)
        elif method == 'log_reg':
            classifier = self.log_reg()
            classifier.fit(self.X_train, self.y_train)
            return classifier.score(self.X_test, self.y_test)

        classifier.fit(self.X_train, self.y_train)
        predicted = classifier.predict(self.X_test)
        if metric == 'accuracy':
            return metrics.accuracy_score(self.y_test, predicted)
        elif metric == 'balanced_accuracy':
            return metrics.balanced_accuracy_score(self.y_test, predicted)
        elif metric == 'average_precision':
            return metrics.average_precision_score(self.y_test, predicted)
        elif metric == 'neg_brier_score':
            return metrics.brier_score_loss(self.y_test, predicted)
        elif metric == 'f1':
            return metrics.f1_score(self.y_test, predicted)
        elif metric == 'roc_auc':
            return metrics.roc_auc_score(self.y_test, predicted)
        else:
            msg = self.u.msg.msg_arg_err("Validate.predict", "metric", metric, ['accuracy',
                                                                                'balanced_accuracy',
                                                                                'average_precision', 'neg_brier_score',
                                                                                'f1', 'roc_auc'])
            self.u.err_p([msg])
            raise Exception(msg)

    def rf(self, n_trees=10):
        n_trees = n_trees if n_trees is not None else 10
        return RandomForestClassifier(n_estimators=n_trees, random_state=self.random_state)

    def log_reg(self):
        return LogisticRegression(random_state=self.random_state)

    def lin_reg(self):
        return LinearRegression()

    def svm(self):
        return SVC(random_state=self.random_state)
