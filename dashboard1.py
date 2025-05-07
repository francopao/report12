import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
import io
from fredapi import Fred
import fear_and_greed
from datetime import datetime
import matplotlib.pyplot as plt
# --------------------------------------
# SCRAPER Y TRANSFORMADOR DE DATOS
# --------------------------------------

@st.cache_data
def obtener_datos_tesoro(periodos):
    all_data = []
    headers = []
    for year in periodos:
        url = f'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value={year}'
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', {'class': 'usa-table views-table views-view-table cols-26'})
            if table:
                headers = [header.text.strip() for header in table.find_all('th')]
                for row in table.find_all('tr')[1:]:
                    cells = [year] + [cell.text.strip() for cell in row.find_all('td')]
                    all_data.append(cells)

    if all_data:
        headers = ['Year'] + headers
        df = pd.DataFrame(all_data, columns=headers)
        df = df.drop(columns=['1.5 Mo'], errors='ignore')
        df = df.apply(lambda x: x.replace('N/A', pd.NA) if x.dtype == "object" else x)
        df = df.dropna(axis=1, how='all')
        df = df.fillna(0)
        for col in df.columns[2:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        return df
    else:
        return pd.DataFrame()

# --------------------------------------
# FUNCIONES FRED
# --------------------------------------
    
def obtener_datos_fred():
    codigos = {
        # Labor Market
        "Total Nonfarm Payrolls": "PAYEMS",
        "Unemployment Rate": "UNRATE",
        "Labor Force Participation Rate": "CIVPART",
        "Job Openings (JOLTS)": "JTSJOL",
        "Average Hourly Earnings (Total Private)": "CES0500000003",
        "U-6 Unemployment Rate": "U6RATE",
        "Quits Rate (JOLTS)": "JTSQUR",

        # Credit/Market
        "Rating AAA": "BAMLC0A1CAAA",
        "Rating AA": "BAMLC0A2CAA",
        "Rating A": "BAMLC0A3CA",
        "Rating BBB": "BAMLC0A4CBBB",
        "BBB o superior": "BAMLC0A0CM",
        "High Yield": "BAMLH0A0HYM2EY",
        "Investment Grade": "BAMLC0A4CBBBEY",
        "Rating AAA ": "BAMLC0A1CAAASYTW",
        "Rating AA ": "BAMLC0A2CAASYTW",
        "Rating A ": "BAMLC0A3CASYTW",
        "Rating BBB ": "BAMLC0A4CBBBSYTW",
        "High Yield ": "BAMLH0A0HYM2SYTW",
        "10-Year Treasury Market Yield ": "DGS10",
        "5-Year Inflation Expectation ": "T5YIFR",
        "2-Year Treasury Market Yield ": "DGS2",
        "Rating AAA Corporate Yield ": "BAMLC0A1CAAAEY",
        
        # YTW bonds to economic zone
        "Global": "BAMLEMUBCRPIUSSYTW",
        "Euro": "BAMLEMEBCRPIESYTW",
        "Latin America": "BAMLEMRLCRPILASYTW",
        "Asia": "BAMLEMRACRPIASIASYTW",
        "EMEA": "BAMLEMRECRPIEMEASYTW",
        
        # Michingan Consumer Sentiment Index - MCSI
        "MSCI": "UMCSENT",
        "Home Purchase Sentiment Index":"HPSI",
        
        # Monetary Policy
        "Inflation Expectation (University of Michigan)": "MICH",
        "CPI":"CPIAUCSL",
        "30-year Breakeven Inflation": "T30YIEM",
        "5-Year Breakeven Inflation":"T5YIE",
        
        # Consumption
        "Retail Sales": "RSXFS",
        "S&P National Home Price Index":"CSUSHPINSA",
        "Personal Consumption Expenditures": "PCE",
        "Total Vehicle Sales": "TOTALSA"

    }
    datos = {}
    fred = Fred(api_key='762e2ee1c8fab5d038ce317929d47226')
    for nombre, codigo in codigos.items():
        serie = fred.get_series(codigo)
        serie.name = nombre
        datos[nombre] = serie
    return datos

def graficar_fred(datos, titulo, series, zoom=False):
    fig = go.Figure()
    for serie in series:
        data = datos[serie].tail(30) if zoom else datos[serie]
        fig.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines', name=serie))
    fig.update_layout(title=titulo, xaxis_title="Fecha", yaxis_title="Valor", template="plotly_white")
    return fig

# --------------------------------------
# STREAMLIT UI
# --------------------------------------

st.set_page_config(layout="wide")
st.image("https://media.licdn.com/dms/image/v2/C4E0BAQHGRK4sbvBk8w/company-logo_200_200/company-logo_200_200/0/1664209061611/decision_capital_eirl_logo?e=2147483647&v=beta&t=dS9RqOZoCN82k_Jqg6JF9Fm7MAQlNUSfIrEuQdLg_qQ", 
         width=200)
st.title("Global Fixed Income Dashboard - Franco Olivares")

tab1, tab2, tab3, tab4 = st.tabs(["Treasury Yields", "US Corporate Bonds", "US Labor Market", "Equity"])

# --------------------------------------
# TAB 1: CURVAS DEL TESORO
# --------------------------------------
with tab1:
    años = st.multiselect("Selecciona año(s):", list(range(2006, 2026)), default=[2025])
    df = obtener_datos_tesoro(años)

    if not df.empty:
        st.success(f"{df.shape[0]} registros obtenidos.")

        fechas = sorted(df["Date"].unique())
        fechas_seleccionadas = st.multiselect("Selecciona una o más fechas para comparar curvas:", fechas[-10:], default=fechas[-3:])

        if "10 Yr" in df.columns and "2 Yr" in df.columns:
            df["Spread 10Y - 2Y"] = df["10 Yr"] - df["2 Yr"]
            st.metric("📉 Spread 10Y - 2Y actual", f"{df['Spread 10Y - 2Y'].iloc[-1]:.2f} %")
            fig_spread = px.line(df, x="Date", y="Spread 10Y - 2Y", title="Evolución del Spread 10Y - 2Y")
            st.plotly_chart(fig_spread, use_container_width=True)

        st.subheader("Comparación de curvas por fecha")
        fig_comparacion = px.line()

        for fecha in fechas_seleccionadas:
            datos_fecha = df[df["Date"] == fecha].iloc[0]
            maturities = df.columns[2:-2]
            tasas = datos_fecha[maturities].values.astype(float)
            fig_comparacion.add_scatter(x=maturities, y=tasas, mode="lines+markers", name=str(fecha.date()))

        fig_comparacion.update_layout(title="Curvas de rendimiento comparadas", xaxis_title="Plazo", yaxis_title="Rendimiento (%)")
        st.plotly_chart(fig_comparacion, use_container_width=True)

        st.subheader("Rendimiento de los bonos del Tesoro a la par")
        df_anim = df.copy()
        df_anim = df_anim.melt(id_vars=["Date"], value_vars=maturities, var_name="Maturity", value_name="Yield")

        fig_anim = px.line(df_anim, x="Maturity", y="Yield", animation_frame=df_anim["Date"].dt.strftime("%Y-%m-%d"),
                        title="Evolución diaria de la curva de rendimiento")
        fig_anim.update_layout(xaxis_title="Plazo", yaxis_title="Rendimiento (%)")
        st.plotly_chart(fig_anim, use_container_width=True)

        st.subheader("Exportar datos")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Yield Curve')
            if "Spread 10Y - 2Y" in df.columns:
                df[['Date', 'Spread 10Y - 2Y']].to_excel(writer, index=False, sheet_name='Spread')

        st.download_button(label="⬇️ Descargar Excel", data=output.getvalue(), file_name="treasury_yield_curve.xlsx")

    else:
        st.warning("No se encontraron datos para los años seleccionados.")

# --------------------------------------
# TAB 2: SPREADS FRED
# --------------------------------------
with tab2:
    st.subheader("Spreads de bonos corporativos ajustado por cualquier opcionalidad(OAS) en USA")

    datos_fred = obtener_datos_fred()
    series_ids = ["Rating AAA", "Rating AA", "Rating A", "Rating BBB", "High Yield"]
    series_ids2 = ["Rating AAA ", "Rating AA ", "Rating A ", "Rating BBB ", "High Yield "]
    if datos_fred:
        fig_fred = graficar_fred(datos_fred, "Spreads de bonos corporativos por calificación", series_ids)
        st.plotly_chart(fig_fred, use_container_width=True)

        st.subheader("Emerging Markets - YTW")
        fig3 = graficar_fred(datos_fred, "ICE BofA Emerging Markets Corporate Plus Index Semi-Annual Yield to Worst", ["Global", "Euro", "Latin America", "Asia", "EMEA"])
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("US Investment Grade vs US High Yield - YTW")
        fig4 = graficar_fred(datos_fred, "ICE BofA US Index Semi-Annual Yield to Worst", ["Rating AAA ", "Rating AA ", "Rating A ", "Rating BBB ", "High Yield "])
        st.plotly_chart(fig4, use_container_width=True)
        
        st.subheader("US bond yields vs. Median of US 5-year inflation expectation")
        fig5 = graficar_fred(datos_fred, "Bonos de alta calidad vs. Inflación esperada a 5 años(media) en USA", ["10-Year Treasury Market Yield ", "2-Year Treasury Market Yield ", "5-Year Inflation Expectation ", "Rating AAA Corporate Yield "])
        st.plotly_chart(fig5, use_container_width=True)        
        
        

# --------------------------------------
# TAB 3: MERCADO LABORAL (FRED)
# --------------------------------------
with tab3:
    st.subheader("Análisis del Mercado Laboral en USA")

    codigos_laborales = {
        "Tasa de desempleo": "UNRATE",
        "Nonfarm Payrolls": "PAYEMS",
        "Ofertas laborales (JOLTS)": "JTSJOL",
        "Renuncias (Quit Rate)": "JTSQUR",
        "Participación laboral": "CIVPART",
        "Initial Claims (ICSA)": "ICSA"
    }

    fred = Fred(api_key='762e2ee1c8fab5d038ce317929d47226')

    @st.cache_data
    def obtener_serie(codigo):
        return fred.get_series(codigo).dropna()

    datos_laborales = {
        nombre: obtener_serie(codigo)
        for nombre, codigo in codigos_laborales.items()
    }

    z_scores = {}
    resumen_tabla = []

    for nombre, serie in datos_laborales.items():
        z = (serie - serie.mean()) / serie.std()
        z_scores[nombre] = z
        actual = serie.iloc[-1]
        promedio = serie.mean()
        desv = serie.std()
        z_actual = z.iloc[-1]
        resumen_tabla.append({
            "Indicador": nombre,
            "Último valor": round(actual, 2),
            "Promedio histórico": round(promedio, 2),
            "Z-Score actual": round(z_actual, 2),
            "Desviación estándar": round(desv, 2)
        })

    df_resumen = pd.DataFrame(resumen_tabla).sort_values("Z-Score actual", ascending=False)

    def formato_numero(x):
        if isinstance(x, (int, float)):
            return f"{x:,.2f}"
        return x

    def semaforo(z):
        if z > 1:
            return "🔴"
        elif z < -1:
            return "🔵"
        else:
            return "⚪️"

    df_resumen["Alerta"] = df_resumen["Z-Score actual"].apply(semaforo)
    columnas_format = ["Último valor", "Promedio histórico", "Z-Score actual", "Desviación estándar"]
    for col in columnas_format:
        df_resumen[col] = df_resumen[col].apply(formato_numero)

    df_resumen = df_resumen[["Alerta", "Indicador"] + columnas_format]

    st.markdown("### Resumen estadístico con alerta visual")
    st.dataframe(df_resumen, use_container_width=True, height=350)

    fig_z = px.bar(df_resumen, x="Z-Score actual", y="Indicador", orientation='h',
                   color="Z-Score actual", color_continuous_scale="RdBu_r",
                   title=" Desviación Estándar de Indicadores respecto a su Media Histórica")
    fig_z.update_layout(height=400, xaxis_title="Z-Score", yaxis_title="", template="plotly_white")
    st.plotly_chart(fig_z, use_container_width=True)

    df_z_all = pd.DataFrame(z_scores)
    df_z_all.index = pd.to_datetime(df_z_all.index)
    df_z_all = df_z_all.resample("M").mean().interpolate()
    df_z_all = df_z_all[df_z_all.index >= "2018"]

    if "Initial Claims (ICSA)" in df_z_all.columns:
        df_z_heatmap = df_z_all.drop(columns=["Initial Claims (ICSA)"])
    else:
        df_z_heatmap = df_z_all.copy()

    fig_heat = px.imshow(df_z_heatmap.T,
                         aspect="auto",
                         color_continuous_scale="RdBu_r",
                         labels=dict(x="Fecha", y="Indicador", color="Z-Score"),
                         title="Mapa de Calor: ¿Qué tan lejos están los indicadores de su media?")
    fig_heat.update_layout(height=500, xaxis_nticks=20)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("### Evolución de Solicitudes por Desempleo en USA (ICSA)")

    icsa = datos_laborales["Initial Claims (ICSA)"]
    z_icsa = (icsa - icsa.mean()) / icsa.std()
    pct_icsa = icsa.pct_change() * 100

    opcion_icsa = st.radio(
        " Métrica a visualizar de ICSA:",
        ["Variación porcentual", "Z-Score", "Nivel absoluto"],
        horizontal=True
    )

    if opcion_icsa == "Variación porcentual":
        serie_base = pct_icsa
        titulo = "Variación Porcentual de Solicitudes Iniciales por Desempleo (ICSA)"
        y_label = "% Variación"
    elif opcion_icsa == "Z-Score":
        serie_base = z_icsa
        titulo = "Z-Score de Solicitudes Iniciales por Desempleo (ICSA)"
        y_label = "Z-Score"
    else:
        serie_base = icsa
        titulo = "Nivel Absoluto de Solicitudes Iniciales por Desempleo (ICSA)"
        y_label = "Solicitudes"

    # Mostrar resumen al lado del gráfico
    st.markdown(f"**Último valor:** {formato_numero(icsa.iloc[-1])} | "
                f"**Promedio histórico:** {formato_numero(icsa.mean())} | "
                f"**Desviación estándar:** {formato_numero(icsa.std())}")

    fecha_min = icsa.index.min().to_pydatetime()
    fecha_max = icsa.index.max().to_pydatetime()
    fecha_defecto_inicio = pd.to_datetime("2018-01-01").to_pydatetime()

    fecha_slider = st.slider(
        "Rango de fechas para visualizar:",
        min_value=fecha_min, max_value=fecha_max,
        value=(fecha_defecto_inicio, fecha_max),
        format="YYYY-MM"
    )

    serie_filtrada = serie_base[(serie_base.index >= fecha_slider[0]) & (serie_base.index <= fecha_slider[1])]

    fig_icsa = px.line(serie_filtrada, title=titulo, labels={"value": y_label, "index": "Fecha"})

    eventos = [
        {"x0": "1973-10-01", "x1": "1974-03-01", "color": "LightGray", "texto": "Crisis OPEP"},
        {"x0": "1980-01-01", "x1": "1982-08-01", "color": "Thistle", "texto": "Volcker Shock"},
        {"x0": "2000-03-01", "x1": "2002-10-01", "color": "LightSalmon", "texto": "Crisis punto-com"},
        {"x0": "2007-12-01", "x1": "2009-06-01", "color": "LightSalmon", "texto": "Lehman Brothers"},
        {"x0": "2018-07-01", "x1": "2020-01-01", "color": "LightBlue", "texto": "Trump's Tariffs"},
        {"x0": "2020-03-01", "x1": "2021-07-01", "color": "LightSalmon", "texto": "COVID-19"},
        {"x0": "2025-01-20", "x1": "2025-06-01", "color": "LightBlue", "texto": "Trump's Tariffs II"}
    ]

    for evento in eventos:
        fig_icsa.add_vrect(
            x0=evento["x0"], x1=evento["x1"],
            fillcolor=evento["color"], opacity=0.15, layer="below", line_width=0,
            annotation_text=evento["texto"], annotation_position="top left"
        )

    fig_icsa.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_icsa, use_container_width=True)
    


# --------------------------------------
# TAB 4: EQUITY 
# --------------------------------------
with tab4:
    st.subheader("Análisis Comparativo de Activos de Equity y Bonos")

    import yfinance as yf

    tickers = {
        "VIX": "^VIX",
        "S&P 500": "^GSPC",
        "ETF TLT (Bonos Largo Plazo)": "TLT",
        "ETF IEF (Bonos Mediano Plazo)": "IEF"
    }

    fecha_inicio = "2018-01-01"

    @st.cache_data
    def descargar_datos(tickers_dict, start):
        series = []
        nombres_validos = []

        for nombre, ticker in tickers_dict.items():
            try:
                datos = yf.download(ticker, start=start)["Close"]
                if not datos.empty:
                    datos.name = nombre
                    series.append(datos)
                    nombres_validos.append(nombre)
                else:
                    st.warning(f"⚠️ No se encontraron datos para {nombre} ({ticker})")
            except Exception as e:
                st.error(f"❌ Error al descargar {nombre} ({ticker}): {e}")

        if not series:
            return pd.DataFrame()

        return pd.concat(series, axis=1, join="inner").rename(columns=dict(zip(nombres_validos, nombres_validos)))

    precios = descargar_datos(tickers, fecha_inicio)

    if precios.empty:
        st.error("No se pudieron descargar datos válidos. Verifica tu conexión o los tickers.")
        st.stop()

    variaciones = precios.pct_change().fillna(0)
    variaciones_acumuladas = (1 + variaciones).cumprod() - 1

    leyenda_info = [
        f"{nombre}: {precios[nombre].iloc[-1]:,.2f} | σ: {precios[nombre].std():,.2f}"
        for nombre in precios.columns
    ]

    fig_equity = px.line(
        variaciones_acumuladas,
        labels={"value": "Variación Acumulada", "index": "Fecha", "variable": "Activo"},
        title="Comparativo: VIX, S&P 500, TLT e IEF (Variación Porcentual Acumulada)"
    )

    fig_equity.update_traces(mode="lines")
    fig_equity.update_layout(
        height=500,
        template="plotly_white",
        legend_title_text="",
        margin=dict(l=30, r=30, t=60, b=40),
        annotations=[
            dict(
                xref="paper", yref="paper",
                x=0.01, y=0.99,
                xanchor="left", yanchor="top",
                align="left",
                text="<br>".join(leyenda_info),
                showarrow=False,
                font=dict(size=12),
                bordercolor="black",
                borderwidth=1,
                bgcolor="white",
                opacity=0.9
            )
        ]
    )

    st.plotly_chart(fig_equity, use_container_width=True)

    # -------------------------------
    # Segundo gráfico - Sectores
    # -------------------------------
    st.subheader("Rentabilidad Acumulada por Sector desde Fecha Elegida")

    tickers_sectoriales = {
        "Tecnología (XLK)": "XLK",
        "Salud (XLV)": "XLV",
        "Finanzas (XLF)": "XLF",
        "Energía (XLE)": "XLE",
        "Consumo Discrecional (XLY)": "XLY",
        "Industriales (XLI)": "XLI",
        "Materiales (XLB)": "XLB",
        "Servicios de Comunicación (XLC)": "XLC",
        "Consumo Básico (XLP)": "XLP",
        "Bienes Raíces (XLRE)": "XLRE",
        "Servicios Públicos (XLU)": "XLU",
        "MSCI Latin America (LTAM.L)": "LTAM.L",
        "MSCI Global": "MSCI"
    }

    fecha_default = datetime(2024, 11, 5)
    fecha_usuario = st.date_input("Selecciona fecha de inicio para la rentabilidad acumulada", value=fecha_default)

    @st.cache_data
    def obtener_precios_sectoriales(tickers_dict, start):
        series = []
        nombres_validos = []

        for nombre, ticker in tickers_dict.items():
            try:
                datos = yf.download(ticker, start=start)["Close"]
                if not datos.empty:
                    datos = datos[datos.index >= pd.to_datetime(start)]
                    if not datos.empty:
                        datos.name = nombre
                        series.append(datos)
                        nombres_validos.append(nombre)
                    else:
                        st.warning(f"⚠️ {nombre} ({ticker}) no tiene datos desde {start}")
                else:
                    st.warning(f"⚠️ {nombre} ({ticker}) no tiene datos descargados")
            except Exception as e:
                st.error(f"❌ Error al descargar {nombre} ({ticker}): {e}")

        if not series:
            return pd.DataFrame()

        df = pd.concat(series, axis=1)
        df.columns = nombres_validos
        return df

    precios_sectoriales = obtener_precios_sectoriales(tickers_sectoriales, fecha_usuario)

    if precios_sectoriales.empty:
        st.error("No se pudieron obtener datos de los sectores.")
        st.stop()

    fechas_disponibles = precios_sectoriales.index
    fecha_base_real = fechas_disponibles[fechas_disponibles >= pd.to_datetime(fecha_usuario)].min()

    if pd.isna(fecha_base_real):
        st.error("No hay datos disponibles después de la fecha seleccionada.")
        st.stop()

    precios_inicio = precios_sectoriales.loc[fecha_base_real]
    precios_final = precios_sectoriales.iloc[-1]

    rentabilidad = ((precios_final - precios_inicio) / precios_inicio) * 100
    rentabilidad = rentabilidad.sort_values()

    colores = rentabilidad.apply(lambda x: "green" if x > 4 else ("gray" if 0 <= x <= 4 else "red"))

    fig_bar = go.Figure()

    fig_bar.add_trace(go.Bar(
        x=rentabilidad.values,
        y=rentabilidad.index,
        orientation='h',
        marker_color=colores,
        text=[f"{x:.2f}%" for x in rentabilidad],
        textposition='auto'
    ))

    fig_bar.update_layout(
        title=f"Rentabilidad Acumulada por Sector desde {fecha_base_real.date()}",
        xaxis_title="Rentabilidad (%)",
        yaxis_title="Sector",
        height=600,
        template="plotly_white",
        margin=dict(l=30, r=30, t=60, b=40)
    )

    st.plotly_chart(fig_bar, use_container_width=True)



    # === GRÁFICO 3 ===

from datetime import datetime, timedelta


st.subheader("ETFs sobre renta fija")

# Definir tickers y fechas
bonos_tickers = ["CEMB", "TLT", "SHY", "EBND", "EMB"]
start_bonos = datetime.today() - timedelta(days=365 * 10)
end_bonos = datetime.today()

# Descargar datos
bonos_data = {"Ticker": [], "Precio Actual": [], "Precio Máximo": [], "Precio Mínimo": []}
for ticker in bonos_tickers:
    df = yf.download(ticker, start=start_bonos, end=end_bonos)
    df = df.dropna()
    bonos_data["Ticker"].append(ticker)
    bonos_data["Precio Actual"].append(df["Close"].iloc[-1])
    bonos_data["Precio Máximo"].append(df["Close"].max())
    bonos_data["Precio Mínimo"].append(df["Close"].min())

df_bonos = pd.DataFrame(bonos_data)

# Crear figura Plotly
fig = go.Figure()

# Dibujar líneas y puntos
for i, row in df_bonos.iterrows():
    ticker = row['Ticker']
    actual = float(row['Precio Actual'])
    maximo = float(row['Precio Máximo'])
    minimo = float(row['Precio Mínimo'])

    # Línea vertical (mínimo a máximo)
    fig.add_trace(go.Scatter(
        x=[ticker, ticker],
        y=[minimo, maximo],
        mode='lines',
        line=dict(color='gray', width=2),
        showlegend=False  # No mostrar en leyenda
    ))

    # Punto máximo
    fig.add_trace(go.Scatter(
        x=[ticker],
        y=[maximo],
        mode='markers+text',
        marker=dict(color='green', size=10, symbol='diamond'),  # Hago el máximo más visible (diamante)
        text=["Máx"],
        textposition="top center",
        name="Máximo",
        showlegend=(i == 0)  # Solo mostrar en primer iteración
    ))

    # Punto mínimo
    fig.add_trace(go.Scatter(
        x=[ticker],
        y=[minimo],
        mode='markers+text',
        marker=dict(color='red', size=10, symbol='diamond'),
        text=["Mín"],
        textposition="bottom center",
        name="Mínimo",
        showlegend=(i == 0)
    ))

    # Punto precio actual
    fig.add_trace(go.Scatter(
        x=[ticker],
        y=[actual],
        mode='markers+text',
        marker=dict(color='blue', size=12, symbol='circle'),  # Cambio a azul fuerte el precio actual
        text=[f"{actual:.1f}"],
        textposition="middle right",
        name="Precio Actual",
        showlegend=(i == 0)
    ))

# Ajustes de layout

fig.update_layout(
    title="Precio en los últimos 10 Años: Actual, Mínimo y Máximo",
    xaxis_title="ETF",
    yaxis_title="Precio (USD)",
    xaxis=dict(type='category'),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    annotations=[
        dict(
            xref='paper', yref='paper',
            x=0, y=-0.1,
            text=(
                "<b>ETF:</b> "
                "EMB: Bonos soberanos y corporativos en EMEs. "
                "EBND: Bonos emitidos en MN de EMEs. "
                "TLT: 20Yr Treasury. "
                "SHY: 1Yr - 3Yr Treasury. "
                "CEMB: Bonos corporativos en EMEs."
            ),
            showarrow=False,
            font=dict(size=10, color="gray"),
            align="left"
        )
    ],
    height=500,  # Subo un poco el alto para que no quede apretado
    margin=dict(l=40, r=40, t=60, b=120),
    template="plotly_white"
)


# Mostrar en Streamlit
st.plotly_chart(fig, use_container_width=True)


    # === GRÁFICO 4 ===

st.subheader("Ranking de Rentabilidad Acumulada de Acciones")



# Lista de acciones a analizar
acciones = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "V", "JNJ", "WMT", "PG", "JPM", "UNH", "HD", "MA", "LLY"]

# Selector de fecha inicial
fecha_inicio_stock = st.date_input(
    "Selecciona la fecha inicial para el cálculo de rentabilidad:",
    value=datetime(2024, 11, 5),
    min_value=datetime(2010, 1, 1),
    max_value=datetime.today()
)


def descargar_datos_acciones(lista_tickers, fecha_inicio):
    datos = yf.download(lista_tickers, start=fecha_inicio)["Close"]
    return datos

# Descargar precios
precios_acciones = descargar_datos_acciones(acciones, fecha_inicio_stock)

if precios_acciones.empty:
    st.error("No se pudieron descargar datos de acciones.")
    st.stop()

# Cálculo de rentabilidad acumulada
variaciones_acciones = precios_acciones.pct_change().fillna(0)
variaciones_acumuladas_acciones = (1 + variaciones_acciones).cumprod() - 1

# Tomar solo el último valor (rentabilidad final acumulada)
rentabilidad_final = variaciones_acumuladas_acciones.iloc[-1].sort_values(ascending=False)

# Crear DataFrame para Plotly
df_rentabilidad = rentabilidad_final.reset_index()
df_rentabilidad.columns = ["Acción", "Rentabilidad"]

# Gráfico de barras horizontales
fig_ranking = px.bar(
    df_rentabilidad,
    x="Rentabilidad",
    y="Acción",
    orientation="h",
    color="Rentabilidad",
    color_continuous_scale="RdYlGn",
    labels={"Rentabilidad": "Rentabilidad Acumulada", "Acción": "Acción"},
    title=f"Rentabilidad Acumulada desde {fecha_inicio_stock.strftime('%Y-%m-%d')}",
    text_auto=".2%",
)

fig_ranking.update_layout(
    height=650,
    template="plotly_white",
    coloraxis_showscale=False,
    margin=dict(l=40, r=40, t=60, b=40),
    yaxis=dict(categoryorder="total ascending"),
)

st.plotly_chart(fig_ranking, use_container_width=True)



    


    # === GRÁFICO 5 ===
st.subheader("Fear and Greed Index")

    # Obtener el objeto de fear_and_greed
output = fear_and_greed.get()
    
    # Acceder a los atributos
value = output.value
description = output.description

    # Crear el gráfico tipo gauge
fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Fear and Greed Index: {description.upper()}"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 25], 'color': "#d62728"},         # Extreme Fear
                {'range': [25, 50], 'color': "#ff7f0e"},        # Fear
                {'range': [50, 75], 'color': "#1f77b4"},        # Greed
                {'range': [75, 100], 'color': "#2ca02c"}        # Extreme Greed
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))

fig.update_layout(height=400)

    # Mostrar el gráfico dentro del Tab
st.plotly_chart(fig, use_container_width=True)
