import os
import kaggle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from fpdf import FPDF
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi
import warnings
warnings.filterwarnings('ignore')

# Configurar el directorio de Kaggle
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('.')

class DataAnalyzer:
    def __init__(self, kaggle_username, kaggle_key):
        """Inicializa el analizador de datos con credenciales de Kaggle"""
        self.data = None
        self.output_path = "output"
        os.makedirs(self.output_path, exist_ok=True)
        
        # Configurar credenciales de Kaggle
        os.environ['KAGGLE_USERNAME'] = kaggle_username
        os.environ['KAGGLE_KEY'] = kaggle_key

    def load_dataset(self):
        """Carga el dataset específico de Kaggle"""
        try:
            with st.spinner("Descargando dataset de Kaggle..."):
                dataset_ref = "uom190346a/sleep-health-and-lifestyle-dataset"
                kaggle.api.dataset_download_files(dataset_ref, path=self.output_path, unzip=True)
                
                self.data = pd.read_csv(os.path.join(self.output_path, "Sleep_health_and_lifestyle_dataset.csv"))
                
                # Eliminar la columna Person ID
                self.data = self.data.drop('Person ID', axis=1)
                
                # Crear un expander para la información del dataset
                with st.expander("📊 Información del Dataset", expanded=True):
                    st.markdown("""
                    ### Descripción del Conjunto de Datos:
                    El Conjunto de Datos sobre Salud y Estilo de Vida del Sueño consta de 400 filas y 13 columnas, 
                    abarcando una amplia variedad de variables relacionadas con el sueño y los hábitos diarios. 
                    Incluye detalles como género, edad, ocupación, duración del sueño, calidad del sueño, 
                    nivel de actividad física, niveles de estrés, categoría de IMC, presión arterial, 
                    frecuencia cardíaca, pasos diarios y la presencia o ausencia de trastornos del sueño.
                    """)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Dimensiones:**", self.data.shape)
                    with col2:
                        st.write("**Variables:**", len(self.data.columns))
                    
                    st.write("**Tipos de variables:**")
                    st.write(self.data.dtypes)
                    
                    st.write("**Primeras 10 filas del dataset:**")
                    st.dataframe(self.data.head(10), use_container_width=True)
                
                return True
        except Exception as e:
            st.error(f"Error al cargar el dataset: {e}")
            return False

    def prepare_data(self):
        """Prepara y limpia los datos"""
        if self.data is None:
            return False

        with st.spinner("Preparando datos..."):
            # Convertir columnas a tipo numérico donde sea posible
            for column in self.data.columns:
                try:
                    self.data[column] = pd.to_numeric(self.data[column])
                except:
                    continue

            # Identificar columnas numéricas
            numeric_columns = self.data.select_dtypes(include=['float64', 'int64']).columns

            # Imputar valores nulos en columnas numéricas
            imputer = SimpleImputer(strategy='mean')
            self.data[numeric_columns] = imputer.fit_transform(self.data[numeric_columns])

            st.success("✅ Datos preparados exitosamente")
            return True

    def perform_eda(self):
        """Realiza el análisis exploratorio de datos"""
        if self.data is None:
            return None

        with st.spinner("Realizando análisis exploratorio..."):
            # Crear directorio para visualizaciones
            viz_path = os.path.join(self.output_path, 'visualizations')
            os.makedirs(viz_path, exist_ok=True)

            # 1. Estadísticas descriptivas
            st.subheader("📈 Estadísticas Descriptivas")
            desc_stats = self.data.describe()
            st.dataframe(desc_stats, use_container_width=True)

            # 2. Histogramas
            st.subheader("📊 Distribución de Variables")
            numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
            
            # Crear selectbox para elegir variable a visualizar
            selected_var = st.selectbox(
                "Seleccione una variable para ver su distribución:",
                numeric_cols
            )
            
            fig = plt.figure(figsize=(10, 6))
            sns.histplot(data=self.data, x=selected_var, kde=True)
            plt.title(f'Distribución de {selected_var}')
            st.pyplot(fig)
            plt.close()

            # Guardar todos los histogramas para el informe
            for col in numeric_cols:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=self.data, x=col, kde=True)
                plt.title(f'Distribución de {col}')
                plt.savefig(os.path.join(viz_path, f'hist_{col}.png'))
                plt.close()

            # 3. Matriz de correlación
            st.subheader("🔄 Matriz de Correlación")
            plt.figure(figsize=(12, 8))
            correlation_matrix = self.data[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Matriz de Correlación')
            plt.tight_layout()
            st.pyplot(plt)
            plt.savefig(os.path.join(viz_path, 'correlation_matrix.png'))
            plt.close()

            return viz_path

    def perform_regressions(self, target_column, model_name):
        """Realiza análisis de regresión basado en el modelo seleccionado"""
        if self.data is None:
            return None

        try:
            # Preparar datos
            numeric_data = self.data.select_dtypes(include=['float64', 'int64'])
            X = numeric_data.drop(columns=[target_column])
            y = numeric_data[target_column]

            # Escalar características
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # División de datos
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # Seleccionar y entrenar el modelo
            if model_name == 'Regresión Lineal':
                model = LinearRegression()
            elif model_name == 'Ridge':
                model = Ridge(alpha=1.0)
            elif model_name == 'Lasso':
                model = Lasso(alpha=1.0)
            else:
                raise ValueError(f"Modelo no reconocido: {model_name}")

            # Entrenar modelo
            model.fit(X_train, y_train)
            
            # Realizar predicciones
            predictions = model.predict(X_test)
            
            # Calcular métricas
            results = {
                'r2': r2_score(y_test, predictions),
                'mse': mean_squared_error(y_test, predictions),
                'predictions': predictions,
                'real_values': y_test
            }

            # Crear gráfico de predicciones vs valores reales
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, predictions, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 
                    'r--', lw=2)
            plt.xlabel('Valores Reales')
            plt.ylabel('Predicciones')
            plt.title(f'Predicciones vs Valores Reales - {model_name}')
            
            # Guardar el gráfico
            plt.savefig(os.path.join(self.output_path, 'predictions_vs_real.png'))
            
            # Mostrar gráfico en Streamlit
            st.pyplot(plt)
            plt.close()

            return results

        except Exception as e:
            st.error(f"Error en el análisis de regresión: {e}")
            return None

    def generate_report(self, regression_results, target_column, model_name):
        """Genera un informe PDF con los resultados del análisis"""
        try:
            pdf = FPDF()
            
            # Configuración de estilo
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # Portada
            pdf.add_page()
            pdf.set_font('Arial', 'B', 24)
            pdf.cell(0, 20, 'Informe de Análisis de Datos', ln=True, align='C')
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 10, f'Fecha: {datetime.now().strftime("%d/%m/%Y")}', ln=True, align='C')
            
            # Información del Dataset
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, '1. Información del Dataset', ln=True)
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 10, f'Dimensiones: {self.data.shape[0]} filas x {self.data.shape[1]} columnas', ln=True)
            pdf.cell(0, 10, 'Columnas:', ln=True)
            for col in self.data.columns:
                pdf.cell(0, 10, f'- {col}: {self.data[col].dtype}', ln=True)
            
            # Estadísticas Descriptivas
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, '2. Estadísticas Descriptivas', ln=True)
            desc_stats = self.data.describe().round(2)
            
            # Convertir estadísticas descriptivas a texto formateado
            stats_text = desc_stats.to_string()
            pdf.set_font('Arial', '', 10)
            for line in stats_text.split('\n'):
                pdf.cell(0, 10, line, ln=True)
            
            # Visualizaciones
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, '3. Visualizaciones', ln=True)
            
            # Matriz de correlación
            pdf.cell(0, 10, 'Matriz de Correlación:', ln=True)
            pdf.image(os.path.join(self.output_path, 'visualizations', 'correlation_matrix.png'), 
                     x=10, y=None, w=190)
            
            # Histogramas
            pdf.add_page()
            pdf.cell(0, 10, 'Histogramas de Variables Numéricas:', ln=True)
            numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                pdf.image(os.path.join(self.output_path, 'visualizations', f'hist_{col}.png'), 
                         x=10, y=None, w=190)
                pdf.add_page()
            
            # Resultados de la Regresión
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, '4. Resultados de la Regresión', ln=True)
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 10, f'Modelo utilizado: {model_name}', ln=True)
            pdf.cell(0, 10, f'Variable objetivo: {target_column}', ln=True)
            pdf.cell(0, 10, f'R2 Score: {regression_results["r2"]:.4f}', ln=True)
            pdf.cell(0, 10, f'Error Cuadrático Medio (MSE): {regression_results["mse"]:.4f}', ln=True)
            
            # Gráfico de Predicciones vs Valores Reales
            pdf.add_page()
            pdf.cell(0, 10, 'Predicciones vs Valores Reales:', ln=True)
            pdf.image(os.path.join(self.output_path, 'predictions_vs_real.png'), 
                     x=10, y=None, w=190)
            
            # Guardar PDF
            report_path = os.path.join(self.output_path, 'informe_analisis.pdf')
            pdf.output(report_path)
            return report_path
            
        except Exception as e:
            st.error(f"Error al generar el informe: {e}")
            return None

def main():
    st.set_page_config(
        page_title="Análisis de Datos de Salud y Sueño",
        page_icon="😴",
        layout="wide"
    )
    
    st.title("😴 Análisis de Datos de Salud y Sueño")
    st.markdown("---")
    
    # Inicializar analizador
    analyzer = DataAnalyzer("andreavduran", "560c3459b619c9fcbed4fb9f1905b44f")
    
    # Cargar y preparar datos
    if analyzer.load_dataset():
        analyzer.prepare_data()
        
        # Realizar EDA
        st.markdown("---")
        st.header("🔍 Análisis Exploratorio de Datos")
        analyzer.perform_eda()
        
        # Análisis de Regresión
        st.markdown("---")
        st.header("📊 Análisis de Regresión")
        
        # Seleccionar variable objetivo
        numeric_cols = analyzer.data.select_dtypes(include=['float64', 'int64']).columns
        target_column = st.selectbox("📌 Seleccione la variable objetivo:", numeric_cols)
        
        # Seleccionar modelo de regresión
        model_name = st.selectbox(
            "🤖 Seleccione el modelo de regresión:",
            ['Regresión Lineal', 'Ridge', 'Lasso']
        )
        
        # Columnas para botones
        col1, col2 = st.columns(2)
        
        # Realizar regresiones
        if col1.button("🚀 Ejecutar Regresión"):
            with st.spinner("Ejecutando análisis de regresión..."):
                regression_results = analyzer.perform_regressions(target_column, model_name)
                if regression_results:
                    st.success("✅ Análisis completado")
                    st.write(f"Resultados de {model_name}:")
                    st.metric("R² Score", f"{regression_results['r2']:.4f}")
                    st.metric("MSE", f"{regression_results['mse']:.4f}")
                    
                    # Guardar los resultados en session_state
                    st.session_state['regression_results'] = regression_results
                    st.session_state['target_column'] = target_column
                    st.session_state['model_name'] = model_name
        
        # Generar informe
        if col2.button("📄 Generar Informe PDF"):
            if 'regression_results' in st.session_state:
                with st.spinner('Generando informe PDF...'):
                    report_path = analyzer.generate_report(
                        st.session_state['regression_results'],
                        st.session_state['target_column'],
                        st.session_state['model_name']
                    )
                    if report_path:
                        st.success("✅ Informe generado exitosamente")
                        
                        # Botón de descarga
                        with open(report_path, "rb") as pdf_file:
                            PDFbyte = pdf_file.read()
                            st.download_button(
                                label="⬇️ Descargar Informe PDF",
                                data=PDFbyte,
                                file_name="informe_analisis.pdf",
                                mime='application/octet-stream'
                            )
            else:
                st.warning("⚠️ Por favor, ejecute primero el análisis de regresión")

if __name__ == "__main__":
    main()