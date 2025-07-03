import os
import json
import csv
from typing import Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_residuals


class ImpedanceAnalysis:
    def __init__(
        self,
        data_dir: str,
        circuit_definition: str = 'R0-p(CPE1,R1-p(R2,L1))',
        initial_guess: Optional[List[float]] = None,
        file_index: int = 0
    ):
        """
        Инициализация анализа импеданса.

        Args:
            data_dir (str): Путь к папке с файлами данных.
            circuit_definition (str): Строка определения схемы эквивалентной цепи.
            initial_guess (Optional[List[float]]]): Начальные параметры для фиттинга.
            file_index (int): Индекс файла для анализа.
        """
        self.data_dir = data_dir
        self.circuit_definition = circuit_definition
        self.initial_guess = initial_guess or [0.5, 1e-6, 1, 45, 3.5, 0.3]
        self.file_index = file_index

        self.frequencies: Optional[np.ndarray] = None
        self.Z: Optional[np.ndarray] = None
        self.circuit: Optional[CustomCircuit] = None
        self.Z_fit: Optional[np.ndarray] = None

        self._load_data()
        self._initialize_circuit()

    def _get_txt_files(self) -> List[str]:
        """Возвращает список .txt файлов в папке."""
        return [file for file in os.listdir(self.data_dir) if file.endswith('.txt')]

    def _load_data(self) -> None:
        """Загружает данные из файла по индексу."""
        txt_files = self._get_txt_files()
        if not txt_files:
            raise FileNotFoundError("Нет .txt файлов в указанной директории.")
        selected_file = txt_files[self.file_index]
        full_path = os.path.join(self.data_dir, selected_file)
        self.frequencies, self.Z = self._read_txt_imp(full_path)
        print(f"Загружен файл: {selected_file}")

    @staticmethod
    def _read_txt_imp(path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Читает данные из файла.

        Args:
            path (str): Путь к файлу.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Частоты и комплексное сопротивление.
        """
        data = np.genfromtxt(path, delimiter=';', skip_header=1)
        freq = data[:, 3]
        Z = np.array([complex(row[4], -row[5]) for row in data])
        return freq, Z

    def _initialize_circuit(self) -> None:
        """Создает объект модели схемы."""
        self.circuit = CustomCircuit(
            circuit=self.circuit_definition,
            initial_guess=self.initial_guess
        )
        print("Модель схемы инициализирована.")

    def fit(self) -> None:
        """Выполняет фиттинг модели к данным."""
        if self.frequencies is None or self.Z is None:
            raise RuntimeError("Данные не загружены.")
        self.circuit.fit(self.frequencies, self.Z)
        self.Z_fit = self.circuit.predict(self.frequencies)
        print("Фиттинг завершен.")
        print(self.circuit)

    def plot_residuals(self, y_limits: Tuple[float, float] = (-10, 10)) -> None:
        """Строит график остатков."""
        if self.frequencies is None or self.Z is None or self.Z_fit is None:
            raise RuntimeError("Данные или модель не готовы.")
        residual_real = (self.Z - self.Z_fit).real / np.abs(self.Z)
        residual_imag = (self.Z - self.Z_fit).imag / np.abs(self.Z)

        fig, ax = plt.subplots()
        plt.tight_layout()
        plot_residuals(ax, self.frequencies, residual_real, residual_imag, y_limits=y_limits)
        plt.show()

    def plot_spectrum(self) -> None:
        """Строит спектр данных и модели."""
        if self.frequencies is None or self.Z is None or self.circuit is None:
            raise RuntimeError("Данные или модель не готовы.")
        self.circuit.plot(f_data=self.frequencies, Z_data=self.Z)

    def run_full_analysis(self) -> None:
        """Запускает полный цикл анализа: фиттинг и визуализация."""
        self.fit()
        self.plot_residuals()
        self.plot_spectrum()

    def export_to_csv(self, filename: str) -> None:
        """
        Экспорт результатов в CSV файл.

        Args:
            filename (str): Имя файла для сохранения.
        """
        if self.frequencies is None or self.Z is None or self.Z_fit is None:
            raise RuntimeError("Нет данных для экспорта.")
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Заголовки
            writer.writerow(['Frequency', 'Z_real', 'Z_imag', 'Z_fit_real', 'Z_fit_imag'])
            for freq, Z, Z_pred in zip(self.frequencies, self.Z, self.Z_fit):
                writer.writerow([freq, Z.real, Z.imag, Z_pred.real, Z_pred.imag])
        print(f"Результаты экспортированы в {filename}")

    def export_to_json(self, filename: str) -> None:
        """
        Экспорт результатов в JSON файл.

        Args:
            filename (str): Имя файла для сохранения.
        """
        if self.frequencies is None or self.Z is None or self.Z_fit is None:
            raise RuntimeError("Нет данных для экспорта.")
        data = {
            'frequencies': self.frequencies.tolist(),
            'Z_real': self.Z.real.tolist(),
            'Z_imag': self.Z.imag.tolist(),
            'Z_fit_real': self.Z_fit.real.tolist(),
            'Z_fit_imag': self.Z_fit.imag.tolist()
        }
        with open(filename, 'w') as jsonfile:
            json.dump(data, jsonfile, indent=4)
        print(f"Результаты экспортированы в {filename}")


if __name__ == "__main__":
    # Пример использования
    analysis = ImpedanceAnalysis(
        data_dir=r'Абсолютный или локальный путь к файлу. Например, D:\Магистр\научка\Импеданс\Mg clear 13-04-21\Exp',
        circuit_definition='R0-p(CPE1,R1-p(R2,L1))',
        initial_guess=[0.5, 1e-6, 1, 45, 3.5, 0.3],
        file_index=0
    )
    analysis.run_full_analysis()

    # Экспорт результатов
    analysis.export_to_csv('results.csv')
    analysis.export_to_json('results.json')