from costcla.models import CostSensitiveDecisionTreeClassifier
from costcla.models import RandomPatchesCostSensitiveDecisionTreeClassifier
from costcla.models import BaggingCostSensitiveDecisionTreeClassifier
from costcla.metrics import savings_score
from sklearn.cross_validation import train_test_split
from costcla.datasets import load_creditscoring1

data = load_creditscoring1()
sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets

# f1 = CostSensitiveDecisionTreeClassifier()
# f1.fit(X_train,y_train, cost_mat_train)
# pred_tree = f1.predict(X_test)
algo = RandomPatchesCostSensitiveDecisionTreeClassifier(combination='majority_voting',
                                        n_estimators=10, verbose=100, pruned=True)
algo.fit(X_train,y_train, cost_mat_train)


f.__setattr__('combination', 'majority_voting')
pred_majority_voting = f.predict(X_test)
f.__setattr__('combination', 'weighted_voting')
pred_weighted_voting = f.predict(X_test)
f.__setattr__('combination', 'stacking')
f._fit_stacking_model(X_train, y_train, cost_mat_train)
pred_stacking = f.predict(X_test)

print savings_score(y_test, pred_tree, cost_mat_test)
print savings_score(y_test, pred_stacking, cost_mat_test)
print savings_score(y_test, pred_majority_voting, cost_mat_test)
print savings_score(y_test, pred_weighted_voting, cost_mat_test)


f = BaggingCostSensitiveDecisionTreeClassifier(combination='stacking',
                                        n_estimators=50, verbose=100,)
f.fit(X_train,y_train, cost_mat_train)


pred_stacking = f.predict(X_test)
f.__setattr__('combination', 'majority_voting')
pred_majority_voting = f.predict(X_test)
f.__setattr__('combination', 'weighted_voting')
pred_weighted_voting = f.predict(X_test)

print savings_score(y_test, pred_tree, cost_mat_test)
print savings_score(y_test, pred_stacking, cost_mat_test)
print savings_score(y_test, pred_majority_voting, cost_mat_test)
print savings_score(y_test, pred_weighted_voting, cost_mat_test)


