import numpy as np
from abc import ABC, abstractmethod
from interfaces import LearningRateSchedule, AbstractOptimizer, LinearRegressionInterface


# ===== Learning Rate Schedules =====
class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        """
        returns: float, learning rate для iteration шага обучения
        """
        return self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p


# ===== Base Optimizer =====
class BaseDescent(AbstractOptimizer, ABC):
    """
    Оптимизатор, имплементирующий градиентный спуск.
    Ответственен только за имплементацию общего алгоритма спуска.
    Все его составные части (learning rate, loss function+regularization) находятся вне зоны ответственности этого класса (см. Single Responsibility Principle).
    """
    def __init__(self, 
                 lr_schedule: LearningRateSchedule = TimeDecayLR(), 
                 tolerance: float = 1e-6,
                 max_iter: int = 1000
                ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """
        Вычисляет обновление согласно конкретному алгоритму и обновляет веса модели, перезаписывая её атрибут.
        Не имеет прямого доступа к вычислению градиента в точке, для подсчета вызывает model.compute_gradients.

        returns: np.ndarray, w_{k+1} - w_k
        """
        pass

    def _step(self) -> np.ndarray:
        """
        Проводит один полный шаг интеративного алгоритма градиентного спуска

        returns: np.ndarray, w_{k+1} - w_k
        """
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self) -> None:
        """
        Оркестрирует весь алгоритм градиентного спуска.
        """
        loss_history = []

        while (self.iteration < self.max_iter):
            loss_history.append(self.model.compute_loss())

            delta = self._step()

            if (np.any(np.isnan(delta))):
                break

            if (np.sum(delta ** 2) < self.tolerance):
                break

        loss_history.append(self.model.compute_loss())

        self.model.loss_history = loss_history


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        gradient = self.model.compute_gradients(X_train, y_train)
        diff = -self.lr_schedule.get_lr(self.iteration + 1) * gradient
        self.model.w += diff
        return diff


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        l = X_train.shape[0]

        mask = np.random.choice(l, self.batch_size, replace = False)
        X_batch = X_train[mask]
        y_batch = y_train[mask]

        gradient = self.model.compute_gradients(X_batch, y_batch)
        diff = -self.lr_schedule.get_lr(self.iteration + 1) * gradient
        self.model.w += diff
        return diff


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.grad_sum = None
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        l, d = X_train.shape

        if self.grad_memory is None:
            self.grad_memory = np.zeros((l, d))
            self.grad_sum = np.zeros(d)

        mask = np.random.choice(l, self.batch_size, replace = False)
        new_gradients_list = []
        for i in mask:
            new_gradients_list.append(self.model.compute_gradients(X_train[i:i+1], y_train[i:i+1]))

        new_gradients = np.array(new_gradients_list)
        old_gradients = self.grad_memory[mask]
        self.grad_sum += np.sum(new_gradients - old_gradients, axis=0)
        self.grad_memory[mask] = new_gradients

        gradient = self.grad_sum / l
        
        diff = -self.lr_schedule.get_lr(self.iteration + 1) * gradient
        self.model.w += diff
        return diff


class MomentumDescent(BaseDescent):
    def __init__(self,  *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        l, d = X_train.shape

        if self.velocity is None:
            self.velocity = np.zeros(d)

        grad = self.model.compute_gradients(X_train, y_train)
        self.velocity *= self.beta
        self.velocity += self.lr_schedule.get_lr(self.iteration) * grad
        self.model.w -= self.velocity
        return -self.velocity

class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        l, d = X_train.shape
        if self.m is None:
            self.m = np.zeros(d)
            self.v = np.zeros(d)
        grad = self.model.compute_gradients(X_train, y_train)
        self.m *= self.beta1
        self.m += (1 - self.beta1) * grad

        self.v *= self.beta2
        self.v += (1 - self.beta2) * (grad ** 2)

        m_mid = self.m / (1 - self.beta1 ** (self.iteration + 1))
        v_mid = self.v / (1 - self.beta2 ** (self.iteration + 1))
        diff = -self.lr_schedule.get_lr(self.iteration + 1) * m_mid
        diff /= (np.sqrt(v_mid) + self.eps)

        self.model.w += diff

        return diff


# ===== Non-iterative Algorithms ====
class AnalyticSolutionOptimizer(AbstractOptimizer):
    """
    Универсальный дамми-класс для вызова аналитических решений 
    """
    def __init__(self):
        self.model = None
    

    def optimize(self) -> None:
        """
        Определяет аналитическое решение и назначает его весам модели.
        """
        # не должна содержать непосредственных формул аналитического решения, за него ответственен другой объект
        X, y = self.model.X_train, self.model.y_train
        loss_func = self.model.loss_function

        self.model.w = loss_func.analytic_solution(X, y)
