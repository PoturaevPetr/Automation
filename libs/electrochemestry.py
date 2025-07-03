import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Optional, Tuple, List, Union

# ================== Potenciodinamic ==================

class Potenciodinamic:
    """Анализ данных потенциодинамических измерений."""

    @staticmethod
    def read_P30(
        name: str,
        S: float = 1.0,
        ref: Optional[float] = None,
        E_p: Optional[float] = None,
        preproces: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Чтение файла данных потенциостата Р30 и распознавание размерностей.
        """
        skip_header = preproces if preproces is not None else 7
        data = np.genfromtxt(name, skip_header=skip_header)

        with open(name, 'r') as f:
            for n, line in enumerate(f, start=1):
                if n == skip_header:
                    header_line = line
                    break
            else:
                header_line = ""

        ref_val = ref if ref is not None else 0
        E_p_val = E_p if E_p is not None else 0

        header_lower = header_line.lower()

        def parse_voltage(val: np.ndarray) -> np.ndarray:
            if 'в' in header_lower:
                return E_p_val - val - ref_val
            elif 'мв' in header_lower:
                return E_p_val - val / 1000 - ref_val
            elif 'мкв' in header_lower:
                return E_p_val - val / 1_000_000 - ref_val
            else:
                return E_p_val - val - ref_val

        def parse_current(val: np.ndarray) -> np.ndarray:
            if 'а' in header_lower:
                return val / S
            elif 'ма' in header_lower:
                return val / (1000 * S)
            elif 'мка' in header_lower:
                return val / (1_000_000 * S)
            else:
                return val / S

        E = parse_voltage(data[:, 1])
        i = parse_current(data[:, 2])

        df = pd.DataFrame({
            'E, B': E,
            'i, A/cm2': i,
            't, c': data[:, 0]
        })

        return df

    @staticmethod
    def cathodic(df: pd.DataFrame) -> pd.DataFrame:
        """
        Выделяет катодную область.
        """
        df_copy = df.copy()
        negative_idx = next((idx for idx, val in enumerate(df_copy['i']) if val < 0), None)
        if negative_idx is not None:
            df_cathodic = df_copy.iloc[negative_idx:].reset_index(drop=True)
        else:
            df_cathodic = df_copy
        df_cathodic.columns = ['E', 'i', 't']
        return df_cathodic

    @staticmethod
    def anodic(df: pd.DataFrame) -> pd.DataFrame:
        """
        Выделяет анодную область.
        """
        df_values = df.values
        n = next((idx for idx, val in enumerate(df_values[:, 1]) if val < 0), None)
        if n is None:
            return pd.DataFrame(columns=['E', 'i', 't'])
        df_anodic = pd.DataFrame(df_values[:n], columns=['E', 'i', 't'])
        return df_anodic

    @staticmethod
    def values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Возвращает полулогарифмические координаты E и lg(i).
        """
        E = -df.iloc[:, 0].values
        i = df.iloc[:, 1].values
        lg_i = np.log10(-i)
        return pd.DataFrame({'-E, B': E, 'lg(i)': lg_i})

    @staticmethod
    def cut_data(
        df: pd.DataFrame,
        start: Optional[float] = None,
        end: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Обрезка данных по диапазону E.
        """
        array = df.values
        E_vals = array[:, 0]
        start_idx = 0
        end_idx = len(E_vals)

        if start is not None:
            start_idx = next((i for i, v in enumerate(E_vals) if v >= start), len(E_vals))
        if end is not None:
            end_idx = next((i for i, v in enumerate(E_vals) if v > end), len(E_vals))
        trimmed = array[start_idx:end_idx]
        result_df = pd.DataFrame(trimmed, columns=['E, B', 'i, A/cm2', 't, c'])
        return result_df

    @staticmethod
    def values_tafel(
        df: pd.DataFrame,
        start: Optional[float] = None,
        end: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Полулогарифмические координаты для участка кривой.
        """
        array = df.values
        E = -array[:, 0]
        i = array[:, 1]
        # Предположим, что i отрицательное
        lg_i = np.log10(np.abs(i))
        # Выбор диапазона
        mask = np.ones(len(E), dtype=bool)
        if start is not None:
            mask &= E >= start
        if end is not None:
            mask &= E <= end
        E_selected = E[mask]
        lg_i_selected = lg_i[mask]
        return pd.DataFrame({'-E, B': E_selected, 'lg(i)': lg_i_selected})

    @staticmethod
    def plot_E_i(df: pd.DataFrame, size: Tuple[int, int] = (15, 10)) -> Axes:
        """
        Построение графика i = f(E).
        """
        fig, ax = plt.subplots(figsize=size)
        E = df.iloc[:, 0].values
        i = df.iloc[:, 1].values
        ax.plot(E, i, '-o')
        ax.set_xlabel('E, В')
        ax.set_ylabel('i, А')
        ax.grid()
        plt.show()
        return ax

    @staticmethod
    def plot_lgi_E(
        df: pd.DataFrame,
        size: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Построение lg(i) = f(E).
        """
        fig, axes = plt.subplots(figsize=size)
        data = df.values
        E = data[:, 0]
        i = data[:, 1]
        i_abs = np.abs(i)
        axes.plot(np.log10(i_abs), E, 'o-')
        axes.set_xscale('log')
        axes.set_xlabel('lg(i), [A/cm$^2$]')
        axes.set_ylabel('-E, В')
        axes.grid()

        # Вставка вспомогательного графика
        inset_ax = fig.add_axes([0.2, 0.55, 0.3, 0.25])
        inset_ax.plot(-E, i_abs, 'o-')
        inset_ax.set_xlabel('E, В')
        inset_ax.set_ylabel('i, А/cm$^2$')
        inset_ax.grid()

        plt.show()
        return fig

    @staticmethod
    def plot_zoom(
        df: pd.DataFrame,
        start: Optional[float] = None,
        end: Optional[float] = None,
        size: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Построение кривой с линией тренда и уравнением Тафеля.
        """
        fig, ax = plt.subplots(figsize=size)
        data = df.values
        E = data[:, 0]
        i = data[:, 1]

        # диапазон
        start_idx = 0
        end_idx = len(E)
        if start is not None:
            start_idx = next((i for i, v in enumerate(E) if v >= start), start_idx)
        if end is not None:
            end_idx = next((i for i, v in enumerate(E) if v > end), end_idx)

        E_range = E[start_idx:end_idx]
        i_range = i[start_idx:end_idx]
        lgi = np.log10(i_range * -1)

        # полиномиальная аппроксимация
        coeffs = np.polyfit(lgi, E_range, 1)
        trend_line = np.poly1d(coeffs)(lgi)
        r2 = r2_score(E_range, trend_line)

        # график
        ax.plot(lgi, E_range, 'o-', label='Данные')
        ax.plot(lgi, trend_line, 'r--', label='Линия тренда')
        ax.legend()

        # уравнение и R^2
        textstr = f'$y={coeffs[0]:.4f}x{coeffs[1]:+.4f}$\n$R^2={r2:.3f}$'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top')

        ax.set_xlabel('lg(i), [A/cm$^2$]')
        ax.set_ylabel('-E, В')
        ax.grid()

        # Вспомогательный график
        inset_ax = fig.add_axes([0.55, 0.2, 0.4, 0.3])
        inset_ax.plot(np.log10(np.abs(i) * -1), E, 'o-')
        inset_ax.set_xlabel('lg(i), [A/cm$^2$]')
        inset_ax.set_ylabel('E, В')
        inset_ax.grid()

        plt.show()
        return fig

    def parameters(
        df: pd.DataFrame,
        real_temp: Optional[float] = None,
        start: Optional[float] = None,
        end: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Расчет параметров по участку кривой.
        """
        data = df.values
        E_vals = data[:, 0]
        I_vals = data[:, 1]
        n = 0
        if start is not None:
            n = next((i for i, v in enumerate(E_vals) if v >= start), 0)
        k = len(E_vals)
        if end is not None:
            k = next((i for i, v in enumerate(E_vals) if v > end), len(E_vals))
        E_seg = E_vals[n:k]
        I_log = np.log10(np.abs(I_vals[n:k]))

        # линейный фит
        coeffs = np.polyfit(I_log, E_seg, 1)
        a, b = coeffs
        i0 = 10 ** (-b / a)
        T = real_temp if real_temp is not None else 298
        kappa = 2.3 * (8.314 * T) / (96485 * a)

        params_df = pd.DataFrame({
            'Parameter': ['$a$', '$b$', '$i_0$', r'$\alpha$'],
            'Value': [b, a, i0, kappa]
        })

        return params_df

    @staticmethod
    def save_dataframe_to_csv(df: pd.DataFrame, filename: str):
        """
        Сохранение DataFrame в CSV.
        """
        df.to_csv(filename, index=False)

    @staticmethod
    def save_plot_as_image(ax: Axes, filename: str, format: str = 'png'):
        """
        Сохранение графика в файл.
        """
        fig = ax.get_figure()
        fig.savefig(filename, format=format)

    @staticmethod
    def process_directory(
        directory: str,
        file_extension: str = '.txt'
    ) -> List[pd.DataFrame]:
        """
        Обработка всех файлов в папке.
        """
        files = [f for f in os.listdir(directory) if f.endswith(file_extension)]
        dataframes = []
        for filename in files:
            filepath = os.path.join(directory, filename)
            try:
                df = Potenciodinamic.read_P30(filepath)
                dataframes.append(df)
                print(f"Обработан файл: {filename}")
            except Exception as e:
                print(f"Ошибка при обработке файла {filename}: {e}")
        return dataframes

    @staticmethod
    def export_results_to_json(results: List[dict], filename: str):
        """
        Экспорт результатов анализа в JSON.
        """
        with open(filename, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

# ================== Cinetic ==================

class Cinetic:
    def parse_Tafel(self, data: np.ndarray) -> None:
        """
        Обработка данных КПК.
        """
        x = data[:, 0]
        y = data[:, 1]
        # Реализуйте обработку как необходимо
        # Например, расчет параметров Тэфеля
        pass

# ================== Potenciostatic ==================

class Potenciostatic:
    @staticmethod
    def list_files(directory: str) -> List[str]:
        """
        Список файлов в директории.
        """
        return os.listdir(directory)

    @staticmethod
    def read_files(name: str, col_points: int = 100) -> Tuple[float, float]:
        """
        Чтение файла и возвращение средних значений потенциала и тока.
        """
        data = np.genfromtxt(name, skip_header=63)
        if data.shape[0] < col_points:
            raise ValueError(f"Недостаточно данных в файле {name}")
        E_mean = float(np.mean(data[:col_points, 0]))
        I_mean = float(np.mean(data[:col_points, 1]))
        return E_mean, I_mean

# ================== Impedance ==================

class Impedance:
    @staticmethod
    def complex_plot(
        ax: Optional[plt.Axes] = None,
        freq: Optional[np.ndarray] = None,
        Z: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        Z_fit: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        label: Optional[str] = None,
        grane_plot: bool = False,
        color: Optional[str] = None,
        projection: Optional[str] = None
    ) -> plt.Axes:
        """
        Построение 3D графика комплексного импеданса.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        if freq is None or Z is None:
            raise ValueError("freq и Z должны быть переданы и не None.")

        Z_prime, Z_double_prime = Z
        ax.plot(Z_prime, np.log10(freq), Z_double_prime, '-o', label=label, linewidth=2)

        if Z_fit is not None:
            Z_fit_prime, Z_fit_double_prime = Z_fit
            ax.plot(Z_fit_prime, np.log10(freq), Z_fit_double_prime, '-o', label=label + ' fit', linewidth=2)

        if grane_plot:
            max_freq_log = np.log10(np.max(freq))
            min_freq_log = np.log10(np.min(freq))
            min_Zp, min_Zpp = np.min(Z_prime), np.min(Z_double_prime)

            # Границы
            ax.plot(Z_prime, np.full_like(freq, max_freq_log), Z_double_prime, color=color)
            ax.plot(np.full_like(Z_prime, min_Zp), np.log10(freq), Z_double_prime, color=color)
            ax.plot(Z_prime, np.log10(freq), np.full_like(Z_double_prime, min_Zpp), color=color)

            # Проекции
            if projection in ('x', 'y', 'z'):
                for i in range(len(freq)):
                    if projection == 'x':
                        ax.plot([Z_prime[i], Z_prime[i]],
                                [np.log10(freq)[i], np.log10(freq)[i]],
                                [Z_double_prime[i], np.min(Z_double_prime)],
                                color='blue')
                    elif projection == 'y':
                        ax.plot([Z_prime[i], Z_prime[i]],
                                [np.log10(freq)[i], np.log10(np.max(freq))],
                                [Z_double_prime[i], Z_double_prime[i]],
                                color='blue')
                    elif projection == 'z':
                        ax.plot([Z_prime[i], np.min(Z_prime)],
                                [np.log10(freq)[i], np.log10(freq)[i]],
                                [Z_double_prime[i], Z_double_prime[i]],
                                color='blue')

        # Подписи
        if label:
            ax.plot(Z_prime, np.log10(freq), Z_double_prime, label=label + (' fit' if Z_fit else ''))
        ax.set_xlabel(r"$Z^{\prime}(\omega)$")
        ax.set_ylabel(r"f [Hz]")
        ax.set_zlabel(r"$Z^{\prime\prime}(\omega)$")
        return ax

# ================== Общие утилиты ==================

def save_dataframe_to_csv(df: pd.DataFrame, filename: str):
    df.to_csv(filename, index=False)

def save_plot_as_image(ax: Axes, filename: str, format: str = 'png'):
    fig = ax.get_figure()
    fig.savefig(filename, format=format)

def process_directory_files(
    directory: str,
    extension: str = '.txt',
    processor_func=None
) -> List[pd.DataFrame]:
    """
    Обработка всех файлов в папке с выбранной функцией обработки.
    """
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    dataframes = []
    for filename in files:
        filepath = os.path.join(directory, filename)
        try:
            if processor_func:
                df = processor_func(filepath)
            else:
                df = pd.read_csv(filepath)
            dataframes.append(df)
            print(f"Обработан файл: {filename}")
        except Exception as e:
            print(f"Ошибка при обработке {filename}: {e}")
    return dataframes

def export_results_to_json(results: List[dict], filename: str):
    with open(filename, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

# ================== Пример использования ==================

if __name__ == "__main__":
    # Обработка нескольких файлов
    directory_path = 'your_data_folder'
    dfs = Potenciodinamic.process_directory(directory_path, extension='.txt')

    # Анализ и экспорт
    results = []
    for df in dfs:
        params = Potenciodinamic.parameters(df)
        results.append(params.to_dict(orient='records'))
        # Можно сохранять графики
        ax = Potenciodinamic.plot_E_i(df)
        filename_img = f"plot_{df.iloc[0,0]:.2f}.png"
        Potenciodinamic.save_plot_as_image(ax, filename_img)

    # Экспорт всех результатов в JSON
    export_results_to_json(results, 'analysis_results.json')