"""Одинаковая шкала оценок в обучении и HTTP-сервисе."""


def spam_scores(model, texts):
    if hasattr(model, "predict_proba"):
        positive = list(model.classes_).index(1)
        return model.predict_proba(texts)[:, positive]
    return model.decision_function(texts)


def default_threshold(model):
    return 0.5 if hasattr(model, "predict_proba") else 0.0
