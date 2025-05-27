# Acquiring the Data

---
## Data Acquisition and Construction of Attention Indexes

This study begins by constructing six thematic Attention Indexes using Google Trends data, each reflecting a distinct retail investor focus in the Taiwanese market: ETFs, individual stocks, dividends, macro-sensitive sectors, technology stocks, and beginner-friendly investments. For each theme, multiple related keywords were selected and queried using the pytrends API. The search volume data for 2024 was normalized and aggregated to form composite weekly indexes that quantify shifts in public interest. These indexes serve as behavioral indicators capturing attention dynamics across different investment mindsets.

To align investor attention with actual market activity, we retrieved weekly trading volumes for 19 representative TWSE-listed stocks using the yfinance library. These stocks were chosen based on their relevance to the attention themes, ensuring consistency between behavioral and market-based data. Trading volumes were normalized, and the resulting dataset was merged with the attention indexes along a weekly time axis to create a unified panel.

This dataset forms the basis for addressing our core research questions:

1. Do changes in public attention precede movements in trading activity (RQ1)?

2. Can attention indexes improve short-term predictive models (RQ2)?

3. Do major external events—such as U.S. Fed meetings or visits by industry leaders—simultaneously affect both investor attention and market engagement (RQ3)?

---


```python
# If you have never used pytrends, you should install it
#!pip install pytrends
import pandas as pd
from pytrends.request import TrendReq
import time
import yfinance as yf
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
```

## Building Subgroup Attention Indexes Using Google Trends

<div class="alert alert-block alert-danger">
<b>Warning:</b>

The following cell (I have turned it into a markdown cell just in case.) might fail if you run it too many times, as pytrends limit requests per IP address. For some reason, I can't get the same exact code to acquire the data, maybe my IP address is blocked by Google Trends. However, you still may get the data if you are careful with the process.
</div>

This section of the research builds a comprehensive picture of retail investor attention in Taiwan by analyzing search behavior from Google Trends. Instead of relying on a single keyword, we group related search terms into thematic clusters—such as ETFs, dividends, macroeconomics, and beginner investing—and create composite "attention indexes" that represent different investor mindsets. These indexes serve as behavioral signals that we can later compare to actual trading activity, test for predictability, and observe under macroeconomic shocks. By capturing multiple dimensions of attention, we aim to better understand how public interest reflects or influences financial market behavior.

```
# Initialize pytrends
pytrends = TrendReq(hl='zh-TW', tz=360)

# Define keyword subgroups
subgroups = {
    "ETF_Attention_Index": ['ETF 投資', '0050', '高股息 ETF', '00878', 'ETF 定期定額'],
    "Stock_Attention_Index": ['投資 股票', '台股 投資', '2330', '台積電', '當沖'],
    "Dividend_Attention_Index": ['高股息', '殖利率', '存股', '金融股', '配息'],
    "Beginner_Attention_Index": ['股票是什麼', '怎麼投資', '證券開戶', '股市新手', '股票入門'],
    "Macro_Attention_Index": ['升息', '通膨', '美國股市', 'FED', '經濟衰退'],
    "Tech_Attention_Index": ['半導體', '台積電', 'AI 投資', '高科技股', 'IC 設計']
}

# Timeframe and location
timeframe = '2024-01-01 2024-12-31'
geo = 'TW'

# Container for results
index_dfs = []

# Loop with 5-second delay
for index_name, keyword_list in subgroups.items():
    try:
        print(f"Fetching: {index_name}...")
        pytrends.build_payload(keyword_list, timeframe=timeframe, geo=geo)
        time.sleep(5)  # Delay to avoid 429 rate limit
        
        df = pytrends.interest_over_time().drop(columns='isPartial')
        df.columns = [col.replace(" ", "_") for col in df.columns]
        
        # Normalize
        df_norm = (df - df.mean()) / df.std()
        df_norm[index_name] = df_norm.mean(axis=1)
        
        index_dfs.append(df_norm[[index_name]])
    except Exception as e:
        print(f"Failed to fetch {index_name}: {e}")
        continue

# Merge all into one DataFrame
attention_index_df = pd.concat(index_dfs, axis=1)

# Show preview
attention_index_df.head()

# Save to Excel
attention_index_df.to_excel('attention_index_data.xlsx')
```

<div class="alert alert-warning">
<b>Message:</b> 
    
In case it doesn't run successfuly, I provided a link to the acquired data. Please check it out, I wouldn't delete it before the Spring semester of 2025 ends. And, if I do, I'm pretty sure that I'll put the `.csv` file in my repository.

</div>

Here is the link: [https://docs.google.com/spreadsheets/d/1TDK94m3D_oqx_hV-NZ5SwGBJXWGo9XmR/edit?usp=sharing&ouid=103068230126415922496&rtpof=true&sd=true](https://docs.google.com/spreadsheets/d/1TDK94m3D_oqx_hV-NZ5SwGBJXWGo9XmR/edit?usp=sharing&ouid=103068230126415922496&rtpof=true&sd=true)

<div class="alert alert-block alert-danger">
<b>Warning:</b>

You need to put the `attention_index_data.xlsx` file in the same folder as this Python script in order for the cell below to run.
</div>


```python
attention_index_df = pd.read_excel('attention_index_data.xlsx', index_col=0)
```

## Merging Weekly Market Volume with Attention Indexes

This step connects behavioral data with actual market behavior. By combining Google Trends-based attention indexes with real-world trading volume, we create a unified dataset that allows us to explore how investor interest aligns with or influences financial activity. This merged view enables descriptive comparisons (e.g., trend co-movement), statistical correlation analysis (RQ1), and predictive modeling (RQ2). It also allows us to examine whether external events like Fed announcements shift both attention and market engagement (RQ3). Aligning these time series on a weekly basis ensures consistency and comparability across all variables.

### Mapping Attention Indexes to Representative TWSE Stocks

To ensure that our stock universe reflects the themes captured by each attention index, we selected representative TWSE stocks for each attention category:

| **Attention Index**          | **Suggested Stocks (TWSE)**                                   | **Rationale**                                       |
|-----------------------------|---------------------------------------------------------------|-----------------------------------------------------|
| `ETF_Attention_Index`        | 0050.TW, 006208.TW, 00878.TW, 00713.TW                        | Large ETFs: broad market, ESG, dividend-heavy       |
| `Stock_Attention_Index`      | 2330.TW, 2303.TW, 2412.TW, 3008.TW                            | Blue-chip, highly followed stocks                   |
| `Dividend_Attention_Index`   | 2881.TW, 2882.TW, 0056.TW, 9917.TW, 1101.TW                   | High-yield financials, dividend ETFs, utilities     |
| `Beginner_Attention_Index`   | 0050.TW, 2884.TW, 2603.TW, 1101.TW                            | Common beginner picks (simple, high volume)         |
| `Macro_Attention_Index`      | 1101.TW, 2603.TW, 1301.TW, 2882.TW, 2308.TW                   | Sensitive to macro shifts (interest, exports)       |
| `Tech_Attention_Index`       | 2330.TW, 2303.TW, 3008.TW, 3034.TW, 2454.TW                   | Semiconductors, optics, electronics                 |

This logic ensures that our volume-based market signals are well-aligned with the **public attention captured in search behavior**, providing a meaningful basis for correlation and predictive analysis.


```python
# Define tickers you care about
tickers = [
    '0050.TW', '006208.TW', '00878.TW', '00713.TW',   # ETF-related
    '2330.TW', '2303.TW', '2412.TW', '3008.TW',       # Stock-following
    '2881.TW', '2882.TW', '0056.TW', '9917.TW', '1101.TW',  # Dividend
    '2884.TW', '2603.TW',                             # Beginner-friendly
    '1301.TW', '2308.TW',                             # Macro-sensitive
    '3034.TW', '2454.TW'                              # Tech-specific
]

start_date = '2024-01-01'
end_date = '2025-01-01'

# Download daily data
prices = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

# Resample weekly volume and normalize
volume_dfs = []
for ticker in tickers:
    vol = prices[ticker]['Volume'].resample('W-SUN').sum()
    vol_norm = (vol - vol.mean()) / vol.std()
    volume_dfs.append(vol_norm.rename(f"{ticker}_Volume_norm"))

# Combine all volumes
volume_df = pd.concat(volume_dfs, axis=1)

# Merge with attention index
merged_df = pd.merge(volume_df, attention_index_df, left_index=True, right_index=True, how='inner')

# Preview merged data
merged_df.head()
```

    YF.download() has changed argument auto_adjust default to True


    [*********************100%***********************]  19 of 19 completed





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0050.TW_Volume_norm</th>
      <th>006208.TW_Volume_norm</th>
      <th>00878.TW_Volume_norm</th>
      <th>00713.TW_Volume_norm</th>
      <th>2330.TW_Volume_norm</th>
      <th>2303.TW_Volume_norm</th>
      <th>2412.TW_Volume_norm</th>
      <th>3008.TW_Volume_norm</th>
      <th>2881.TW_Volume_norm</th>
      <th>2882.TW_Volume_norm</th>
      <th>...</th>
      <th>1301.TW_Volume_norm</th>
      <th>2308.TW_Volume_norm</th>
      <th>3034.TW_Volume_norm</th>
      <th>2454.TW_Volume_norm</th>
      <th>ETF_Attention_Index</th>
      <th>Stock_Attention_Index</th>
      <th>Dividend_Attention_Index</th>
      <th>Beginner_Attention_Index</th>
      <th>Macro_Attention_Index</th>
      <th>Tech_Attention_Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024-01-07</th>
      <td>-0.860674</td>
      <td>-0.932212</td>
      <td>-0.379417</td>
      <td>-0.874601</td>
      <td>-1.254108</td>
      <td>-0.166771</td>
      <td>-0.911028</td>
      <td>0.418549</td>
      <td>-1.167018</td>
      <td>-1.203771</td>
      <td>...</td>
      <td>-1.403838</td>
      <td>-1.300789</td>
      <td>-0.466500</td>
      <td>0.439787</td>
      <td>-1.216050</td>
      <td>-1.367152</td>
      <td>-0.199133</td>
      <td>-0.899066</td>
      <td>0.390401</td>
      <td>-0.575975</td>
    </tr>
    <tr>
      <th>2024-01-14</th>
      <td>-0.653673</td>
      <td>-0.931290</td>
      <td>-0.708484</td>
      <td>-0.822152</td>
      <td>-1.295240</td>
      <td>-0.650347</td>
      <td>-0.810122</td>
      <td>1.288575</td>
      <td>-0.906104</td>
      <td>-0.998040</td>
      <td>...</td>
      <td>-0.842457</td>
      <td>-0.264088</td>
      <td>-0.352294</td>
      <td>-0.322731</td>
      <td>-0.116791</td>
      <td>-0.582552</td>
      <td>0.236205</td>
      <td>-0.666854</td>
      <td>-0.072779</td>
      <td>-0.180896</td>
    </tr>
    <tr>
      <th>2024-01-21</th>
      <td>1.345122</td>
      <td>-0.478630</td>
      <td>-0.280000</td>
      <td>-0.611415</td>
      <td>1.391158</td>
      <td>0.676357</td>
      <td>-0.147496</td>
      <td>0.498769</td>
      <td>0.186507</td>
      <td>-0.045758</td>
      <td>...</td>
      <td>0.223543</td>
      <td>0.332889</td>
      <td>1.014547</td>
      <td>0.856694</td>
      <td>-0.433229</td>
      <td>-0.582552</td>
      <td>-0.121921</td>
      <td>-0.173195</td>
      <td>-0.007083</td>
      <td>-0.180896</td>
    </tr>
    <tr>
      <th>2024-01-28</th>
      <td>0.804586</td>
      <td>-0.368364</td>
      <td>-0.302921</td>
      <td>-0.923418</td>
      <td>0.623496</td>
      <td>1.909554</td>
      <td>-0.861677</td>
      <td>-0.341718</td>
      <td>-0.874992</td>
      <td>-1.063764</td>
      <td>...</td>
      <td>-0.901903</td>
      <td>-0.248834</td>
      <td>-0.524553</td>
      <td>0.512786</td>
      <td>-0.119907</td>
      <td>-1.067593</td>
      <td>-0.194599</td>
      <td>-0.563430</td>
      <td>-0.396802</td>
      <td>-0.685401</td>
    </tr>
    <tr>
      <th>2024-02-04</th>
      <td>0.260986</td>
      <td>-0.821956</td>
      <td>-0.410453</td>
      <td>-1.048354</td>
      <td>-0.096706</td>
      <td>-0.126591</td>
      <td>-0.394153</td>
      <td>0.253704</td>
      <td>-0.933695</td>
      <td>-0.997090</td>
      <td>...</td>
      <td>-0.971914</td>
      <td>-0.339969</td>
      <td>-0.359664</td>
      <td>1.502433</td>
      <td>-1.383650</td>
      <td>-2.106888</td>
      <td>-1.136914</td>
      <td>-0.922065</td>
      <td>-0.981105</td>
      <td>-1.908722</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
merged_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0050.TW_Volume_norm</th>
      <th>006208.TW_Volume_norm</th>
      <th>00878.TW_Volume_norm</th>
      <th>00713.TW_Volume_norm</th>
      <th>2330.TW_Volume_norm</th>
      <th>2303.TW_Volume_norm</th>
      <th>2412.TW_Volume_norm</th>
      <th>3008.TW_Volume_norm</th>
      <th>2881.TW_Volume_norm</th>
      <th>2882.TW_Volume_norm</th>
      <th>...</th>
      <th>1301.TW_Volume_norm</th>
      <th>2308.TW_Volume_norm</th>
      <th>3034.TW_Volume_norm</th>
      <th>2454.TW_Volume_norm</th>
      <th>ETF_Attention_Index</th>
      <th>Stock_Attention_Index</th>
      <th>Dividend_Attention_Index</th>
      <th>Beginner_Attention_Index</th>
      <th>Macro_Attention_Index</th>
      <th>Tech_Attention_Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024-01-07</th>
      <td>-0.860674</td>
      <td>-0.932212</td>
      <td>-0.379417</td>
      <td>-0.874601</td>
      <td>-1.254108</td>
      <td>-0.166771</td>
      <td>-0.911028</td>
      <td>0.418549</td>
      <td>-1.167018</td>
      <td>-1.203771</td>
      <td>...</td>
      <td>-1.403838</td>
      <td>-1.300789</td>
      <td>-0.466500</td>
      <td>0.439787</td>
      <td>-1.216050</td>
      <td>-1.367152</td>
      <td>-0.199133</td>
      <td>-0.899066</td>
      <td>0.390401</td>
      <td>-0.575975</td>
    </tr>
    <tr>
      <th>2024-01-14</th>
      <td>-0.653673</td>
      <td>-0.931290</td>
      <td>-0.708484</td>
      <td>-0.822152</td>
      <td>-1.295240</td>
      <td>-0.650347</td>
      <td>-0.810122</td>
      <td>1.288575</td>
      <td>-0.906104</td>
      <td>-0.998040</td>
      <td>...</td>
      <td>-0.842457</td>
      <td>-0.264088</td>
      <td>-0.352294</td>
      <td>-0.322731</td>
      <td>-0.116791</td>
      <td>-0.582552</td>
      <td>0.236205</td>
      <td>-0.666854</td>
      <td>-0.072779</td>
      <td>-0.180896</td>
    </tr>
    <tr>
      <th>2024-01-21</th>
      <td>1.345122</td>
      <td>-0.478630</td>
      <td>-0.280000</td>
      <td>-0.611415</td>
      <td>1.391158</td>
      <td>0.676357</td>
      <td>-0.147496</td>
      <td>0.498769</td>
      <td>0.186507</td>
      <td>-0.045758</td>
      <td>...</td>
      <td>0.223543</td>
      <td>0.332889</td>
      <td>1.014547</td>
      <td>0.856694</td>
      <td>-0.433229</td>
      <td>-0.582552</td>
      <td>-0.121921</td>
      <td>-0.173195</td>
      <td>-0.007083</td>
      <td>-0.180896</td>
    </tr>
    <tr>
      <th>2024-01-28</th>
      <td>0.804586</td>
      <td>-0.368364</td>
      <td>-0.302921</td>
      <td>-0.923418</td>
      <td>0.623496</td>
      <td>1.909554</td>
      <td>-0.861677</td>
      <td>-0.341718</td>
      <td>-0.874992</td>
      <td>-1.063764</td>
      <td>...</td>
      <td>-0.901903</td>
      <td>-0.248834</td>
      <td>-0.524553</td>
      <td>0.512786</td>
      <td>-0.119907</td>
      <td>-1.067593</td>
      <td>-0.194599</td>
      <td>-0.563430</td>
      <td>-0.396802</td>
      <td>-0.685401</td>
    </tr>
    <tr>
      <th>2024-02-04</th>
      <td>0.260986</td>
      <td>-0.821956</td>
      <td>-0.410453</td>
      <td>-1.048354</td>
      <td>-0.096706</td>
      <td>-0.126591</td>
      <td>-0.394153</td>
      <td>0.253704</td>
      <td>-0.933695</td>
      <td>-0.997090</td>
      <td>...</td>
      <td>-0.971914</td>
      <td>-0.339969</td>
      <td>-0.359664</td>
      <td>1.502433</td>
      <td>-1.383650</td>
      <td>-2.106888</td>
      <td>-1.136914</td>
      <td>-0.922065</td>
      <td>-0.981105</td>
      <td>-1.908722</td>
    </tr>
    <tr>
      <th>2024-02-11</th>
      <td>-1.237189</td>
      <td>-1.421590</td>
      <td>-1.210208</td>
      <td>-1.549736</td>
      <td>-2.016167</td>
      <td>-1.840485</td>
      <td>-1.260164</td>
      <td>-1.655890</td>
      <td>-1.985529</td>
      <td>-1.710400</td>
      <td>...</td>
      <td>-1.739475</td>
      <td>-1.640894</td>
      <td>-1.173451</td>
      <td>-2.052232</td>
      <td>0.302036</td>
      <td>-1.166479</td>
      <td>-0.272490</td>
      <td>0.318262</td>
      <td>-0.596347</td>
      <td>-1.338371</td>
    </tr>
    <tr>
      <th>2024-02-18</th>
      <td>0.229564</td>
      <td>-0.832065</td>
      <td>-0.537091</td>
      <td>-1.107962</td>
      <td>-0.377225</td>
      <td>-1.507081</td>
      <td>-0.955978</td>
      <td>-1.366078</td>
      <td>-1.261262</td>
      <td>-1.515176</td>
      <td>...</td>
      <td>-1.460380</td>
      <td>-0.833227</td>
      <td>-0.787618</td>
      <td>-1.214181</td>
      <td>1.036764</td>
      <td>-0.252119</td>
      <td>0.847305</td>
      <td>1.490656</td>
      <td>-0.423384</td>
      <td>0.103803</td>
    </tr>
    <tr>
      <th>2024-02-25</th>
      <td>0.397291</td>
      <td>-0.774026</td>
      <td>0.910396</td>
      <td>-0.728668</td>
      <td>-0.122954</td>
      <td>-0.556364</td>
      <td>-0.534191</td>
      <td>0.241159</td>
      <td>-0.829691</td>
      <td>-0.791305</td>
      <td>...</td>
      <td>-1.161329</td>
      <td>-0.688308</td>
      <td>3.183674</td>
      <td>1.587759</td>
      <td>0.874391</td>
      <td>-0.969907</td>
      <td>0.603374</td>
      <td>0.844856</td>
      <td>-0.450444</td>
      <td>0.016008</td>
    </tr>
    <tr>
      <th>2024-03-03</th>
      <td>-0.747691</td>
      <td>-0.919958</td>
      <td>1.365247</td>
      <td>-0.988321</td>
      <td>-0.632338</td>
      <td>-0.789527</td>
      <td>0.196394</td>
      <td>0.035029</td>
      <td>-0.779160</td>
      <td>-0.780959</td>
      <td>...</td>
      <td>-1.144995</td>
      <td>-0.850699</td>
      <td>1.115441</td>
      <td>1.559844</td>
      <td>2.139901</td>
      <td>0.686122</td>
      <td>1.814502</td>
      <td>2.460722</td>
      <td>0.219620</td>
      <td>0.827795</td>
    </tr>
    <tr>
      <th>2024-03-10</th>
      <td>2.194655</td>
      <td>-0.066304</td>
      <td>1.156469</td>
      <td>-0.167821</td>
      <td>2.501074</td>
      <td>1.942874</td>
      <td>-0.515357</td>
      <td>-0.659940</td>
      <td>-0.426118</td>
      <td>-0.239959</td>
      <td>...</td>
      <td>-0.143020</td>
      <td>1.376633</td>
      <td>1.914310</td>
      <td>2.459762</td>
      <td>2.722685</td>
      <td>0.597361</td>
      <td>2.618912</td>
      <td>3.073093</td>
      <td>0.181354</td>
      <td>0.498563</td>
    </tr>
    <tr>
      <th>2024-03-17</th>
      <td>0.161742</td>
      <td>-0.668163</td>
      <td>0.362380</td>
      <td>-0.574319</td>
      <td>1.328536</td>
      <td>1.374623</td>
      <td>-0.085341</td>
      <td>-0.904853</td>
      <td>0.913319</td>
      <td>2.413071</td>
      <td>...</td>
      <td>-0.456619</td>
      <td>0.598802</td>
      <td>0.535448</td>
      <td>0.619906</td>
      <td>1.314797</td>
      <td>0.531514</td>
      <td>1.257634</td>
      <td>1.218891</td>
      <td>1.865981</td>
      <td>0.257444</td>
    </tr>
    <tr>
      <th>2024-03-24</th>
      <td>0.366868</td>
      <td>-0.585066</td>
      <td>-0.095628</td>
      <td>-0.544844</td>
      <td>0.122906</td>
      <td>1.571410</td>
      <td>0.113501</td>
      <td>-0.428182</td>
      <td>0.031801</td>
      <td>0.399435</td>
      <td>...</td>
      <td>-0.371602</td>
      <td>3.906885</td>
      <td>1.385474</td>
      <td>0.145393</td>
      <td>0.094737</td>
      <td>-0.043489</td>
      <td>0.160353</td>
      <td>0.526760</td>
      <td>0.358817</td>
      <td>-0.181214</td>
    </tr>
    <tr>
      <th>2024-03-31</th>
      <td>-0.868724</td>
      <td>-0.711998</td>
      <td>-0.466922</td>
      <td>-0.844189</td>
      <td>-0.964132</td>
      <td>0.334022</td>
      <td>0.395530</td>
      <td>-1.144939</td>
      <td>-0.852363</td>
      <td>-0.770476</td>
      <td>...</td>
      <td>-0.935602</td>
      <td>1.250226</td>
      <td>-0.146757</td>
      <td>-0.353166</td>
      <td>-0.716168</td>
      <td>-0.695430</td>
      <td>-0.214125</td>
      <td>-0.660166</td>
      <td>-0.447007</td>
      <td>-1.250893</td>
    </tr>
    <tr>
      <th>2024-04-07</th>
      <td>-1.233640</td>
      <td>-1.123587</td>
      <td>-1.103072</td>
      <td>-1.349250</td>
      <td>-1.351546</td>
      <td>-1.161538</td>
      <td>-0.653474</td>
      <td>-1.562211</td>
      <td>-1.431667</td>
      <td>-1.175307</td>
      <td>...</td>
      <td>-1.067138</td>
      <td>-0.759074</td>
      <td>-1.159734</td>
      <td>-1.414289</td>
      <td>-0.044779</td>
      <td>-0.064471</td>
      <td>0.549143</td>
      <td>0.221729</td>
      <td>0.372724</td>
      <td>0.213547</td>
    </tr>
    <tr>
      <th>2024-04-14</th>
      <td>-0.818413</td>
      <td>-0.740354</td>
      <td>-0.687886</td>
      <td>-1.049827</td>
      <td>-0.144444</td>
      <td>-0.010575</td>
      <td>-0.362846</td>
      <td>0.567848</td>
      <td>-0.297129</td>
      <td>0.519136</td>
      <td>...</td>
      <td>-0.627376</td>
      <td>0.006956</td>
      <td>0.582534</td>
      <td>-0.471792</td>
      <td>0.007389</td>
      <td>0.596160</td>
      <td>0.387149</td>
      <td>0.493112</td>
      <td>0.144458</td>
      <td>0.196457</td>
    </tr>
    <tr>
      <th>2024-04-21</th>
      <td>0.571851</td>
      <td>0.594104</td>
      <td>0.855394</td>
      <td>-0.250078</td>
      <td>1.654554</td>
      <td>0.472620</td>
      <td>0.946073</td>
      <td>-0.294573</td>
      <td>0.520381</td>
      <td>0.074287</td>
      <td>...</td>
      <td>-0.323876</td>
      <td>0.759546</td>
      <td>0.474981</td>
      <td>1.672632</td>
      <td>-0.335291</td>
      <td>0.409712</td>
      <td>0.064673</td>
      <td>0.106817</td>
      <td>-0.082712</td>
      <td>0.213547</td>
    </tr>
    <tr>
      <th>2024-04-28</th>
      <td>-0.694860</td>
      <td>-0.494137</td>
      <td>-0.391587</td>
      <td>-0.846492</td>
      <td>-0.082493</td>
      <td>-0.051680</td>
      <td>0.535171</td>
      <td>-0.798103</td>
      <td>-0.484098</td>
      <td>-0.449243</td>
      <td>...</td>
      <td>-0.707317</td>
      <td>0.010277</td>
      <td>0.082801</td>
      <td>0.151501</td>
      <td>-0.128214</td>
      <td>-0.561570</td>
      <td>0.432994</td>
      <td>-0.415583</td>
      <td>0.584313</td>
      <td>-0.225112</td>
    </tr>
    <tr>
      <th>2024-05-05</th>
      <td>-0.997882</td>
      <td>-1.025244</td>
      <td>-0.197541</td>
      <td>-1.189926</td>
      <td>-0.680383</td>
      <td>-0.667152</td>
      <td>-0.419523</td>
      <td>-1.205714</td>
      <td>-0.644885</td>
      <td>0.112103</td>
      <td>...</td>
      <td>-1.028617</td>
      <td>0.389197</td>
      <td>-0.124997</td>
      <td>-0.426174</td>
      <td>-0.186676</td>
      <td>-0.417819</td>
      <td>0.521743</td>
      <td>-0.091987</td>
      <td>-0.425901</td>
      <td>-0.159265</td>
    </tr>
    <tr>
      <th>2024-05-12</th>
      <td>-0.598918</td>
      <td>-0.904430</td>
      <td>0.398606</td>
      <td>-0.878894</td>
      <td>-0.694490</td>
      <td>-0.739453</td>
      <td>0.070423</td>
      <td>-1.282429</td>
      <td>0.424869</td>
      <td>1.434858</td>
      <td>...</td>
      <td>-0.904875</td>
      <td>-0.358670</td>
      <td>1.443819</td>
      <td>-0.094508</td>
      <td>0.403395</td>
      <td>0.244013</td>
      <td>0.329863</td>
      <td>0.485980</td>
      <td>-0.178401</td>
      <td>-0.049521</td>
    </tr>
    <tr>
      <th>2024-05-19</th>
      <td>-0.159807</td>
      <td>-0.592757</td>
      <td>1.289540</td>
      <td>-0.722975</td>
      <td>-0.146525</td>
      <td>-0.527439</td>
      <td>-0.290846</td>
      <td>-0.849415</td>
      <td>1.944431</td>
      <td>2.234318</td>
      <td>...</td>
      <td>-0.856875</td>
      <td>-0.678815</td>
      <td>0.619120</td>
      <td>0.191536</td>
      <td>-0.095949</td>
      <td>0.178166</td>
      <td>-0.369586</td>
      <td>0.121053</td>
      <td>-0.467470</td>
      <td>-0.115368</td>
    </tr>
    <tr>
      <th>2024-05-26</th>
      <td>-0.891158</td>
      <td>-0.685765</td>
      <td>0.007798</td>
      <td>-0.479959</td>
      <td>-0.386389</td>
      <td>1.726971</td>
      <td>0.851108</td>
      <td>-0.882647</td>
      <td>-0.220710</td>
      <td>-0.007944</td>
      <td>...</td>
      <td>-0.535214</td>
      <td>-0.251497</td>
      <td>1.054751</td>
      <td>-0.632310</td>
      <td>-0.216038</td>
      <td>0.078313</td>
      <td>-0.057608</td>
      <td>0.562499</td>
      <td>-0.070447</td>
      <td>-0.137316</td>
    </tr>
    <tr>
      <th>2024-06-02</th>
      <td>-0.598587</td>
      <td>-0.195215</td>
      <td>-0.083075</td>
      <td>-0.622463</td>
      <td>0.257413</td>
      <td>1.860779</td>
      <td>4.121217</td>
      <td>-0.432042</td>
      <td>0.502147</td>
      <td>0.963097</td>
      <td>...</td>
      <td>0.037962</td>
      <td>1.191590</td>
      <td>2.572330</td>
      <td>1.924596</td>
      <td>0.269329</td>
      <td>0.266928</td>
      <td>0.422328</td>
      <td>0.853964</td>
      <td>0.129664</td>
      <td>0.564410</td>
    </tr>
    <tr>
      <th>2024-06-09</th>
      <td>-0.259398</td>
      <td>-0.226086</td>
      <td>-0.254391</td>
      <td>0.446489</td>
      <td>0.135008</td>
      <td>0.065011</td>
      <td>1.455092</td>
      <td>-0.516794</td>
      <td>-0.160634</td>
      <td>0.025519</td>
      <td>...</td>
      <td>0.199797</td>
      <td>0.290215</td>
      <td>0.535164</td>
      <td>0.229678</td>
      <td>0.071616</td>
      <td>0.057331</td>
      <td>0.330030</td>
      <td>0.509255</td>
      <td>0.364016</td>
      <td>0.257444</td>
    </tr>
    <tr>
      <th>2024-06-16</th>
      <td>-0.407872</td>
      <td>-0.467055</td>
      <td>-0.591574</td>
      <td>0.436613</td>
      <td>0.106846</td>
      <td>-0.285371</td>
      <td>-0.228196</td>
      <td>0.719454</td>
      <td>-0.300659</td>
      <td>0.236267</td>
      <td>...</td>
      <td>0.146745</td>
      <td>0.050750</td>
      <td>-0.435587</td>
      <td>0.099218</td>
      <td>0.380183</td>
      <td>1.072511</td>
      <td>0.577997</td>
      <td>1.072301</td>
      <td>-0.016240</td>
      <td>0.564727</td>
    </tr>
    <tr>
      <th>2024-06-23</th>
      <td>0.306488</td>
      <td>0.308618</td>
      <td>-0.133506</td>
      <td>2.838734</td>
      <td>1.300237</td>
      <td>-0.078119</td>
      <td>0.163321</td>
      <td>1.498530</td>
      <td>-0.002245</td>
      <td>0.717382</td>
      <td>...</td>
      <td>2.026165</td>
      <td>2.167035</td>
      <td>0.288463</td>
      <td>0.758336</td>
      <td>0.085901</td>
      <td>0.288877</td>
      <td>-0.077981</td>
      <td>0.647612</td>
      <td>-0.373478</td>
      <td>-0.286099</td>
    </tr>
    <tr>
      <th>2024-06-30</th>
      <td>-0.252749</td>
      <td>0.048045</td>
      <td>-0.588220</td>
      <td>1.180550</td>
      <td>0.932319</td>
      <td>0.254048</td>
      <td>0.315737</td>
      <td>0.918579</td>
      <td>0.166662</td>
      <td>1.127817</td>
      <td>...</td>
      <td>2.236418</td>
      <td>1.071843</td>
      <td>0.419884</td>
      <td>0.319041</td>
      <td>0.457731</td>
      <td>0.588436</td>
      <td>0.869882</td>
      <td>0.719746</td>
      <td>-0.153498</td>
      <td>0.476932</td>
    </tr>
    <tr>
      <th>2024-07-07</th>
      <td>-0.388168</td>
      <td>-0.091450</td>
      <td>-0.723933</td>
      <td>0.428019</td>
      <td>-0.751663</td>
      <td>2.830180</td>
      <td>3.412728</td>
      <td>0.822942</td>
      <td>0.852822</td>
      <td>2.389287</td>
      <td>...</td>
      <td>-0.227947</td>
      <td>0.135417</td>
      <td>1.024232</td>
      <td>-0.046292</td>
      <td>0.495087</td>
      <td>1.347954</td>
      <td>1.097358</td>
      <td>0.996424</td>
      <td>0.088138</td>
      <td>0.240354</td>
    </tr>
    <tr>
      <th>2024-07-14</th>
      <td>0.389082</td>
      <td>1.774662</td>
      <td>-0.583508</td>
      <td>1.125041</td>
      <td>0.872045</td>
      <td>0.950151</td>
      <td>0.822148</td>
      <td>3.369518</td>
      <td>2.330583</td>
      <td>2.010759</td>
      <td>...</td>
      <td>-0.062726</td>
      <td>1.516642</td>
      <td>0.806194</td>
      <td>0.311527</td>
      <td>0.521328</td>
      <td>2.342152</td>
      <td>0.881700</td>
      <td>0.528625</td>
      <td>0.200632</td>
      <td>1.464310</td>
    </tr>
    <tr>
      <th>2024-07-21</th>
      <td>0.664042</td>
      <td>1.437313</td>
      <td>0.157781</td>
      <td>0.986948</td>
      <td>1.677398</td>
      <td>0.057808</td>
      <td>1.366704</td>
      <td>1.100900</td>
      <td>2.953572</td>
      <td>1.390850</td>
      <td>...</td>
      <td>1.591491</td>
      <td>0.197865</td>
      <td>0.134685</td>
      <td>-0.070990</td>
      <td>-0.642950</td>
      <td>0.520423</td>
      <td>-0.863570</td>
      <td>-0.179684</td>
      <td>-0.437294</td>
      <td>-0.132458</td>
    </tr>
    <tr>
      <th>2024-07-28</th>
      <td>0.238835</td>
      <td>0.644611</td>
      <td>0.016375</td>
      <td>0.047797</td>
      <td>0.421956</td>
      <td>-0.511570</td>
      <td>0.114192</td>
      <td>-0.437656</td>
      <td>-0.060742</td>
      <td>-0.249534</td>
      <td>...</td>
      <td>-0.566979</td>
      <td>-0.093172</td>
      <td>-0.702871</td>
      <td>-0.581049</td>
      <td>0.452443</td>
      <td>0.863880</td>
      <td>0.248797</td>
      <td>-0.275065</td>
      <td>0.669848</td>
      <td>1.200290</td>
    </tr>
    <tr>
      <th>2024-08-04</th>
      <td>0.804270</td>
      <td>0.700778</td>
      <td>1.335979</td>
      <td>0.400453</td>
      <td>0.909105</td>
      <td>1.009556</td>
      <td>0.724196</td>
      <td>0.889059</td>
      <td>0.563817</td>
      <td>0.020863</td>
      <td>...</td>
      <td>-0.518163</td>
      <td>1.587337</td>
      <td>0.118552</td>
      <td>0.940821</td>
      <td>1.678034</td>
      <td>2.775571</td>
      <td>1.100414</td>
      <td>0.360503</td>
      <td>1.589670</td>
      <td>1.336841</td>
    </tr>
    <tr>
      <th>2024-08-11</th>
      <td>5.111937</td>
      <td>4.302629</td>
      <td>3.125536</td>
      <td>2.302897</td>
      <td>3.050993</td>
      <td>0.852428</td>
      <td>1.852844</td>
      <td>0.941820</td>
      <td>1.707335</td>
      <td>1.721620</td>
      <td>...</td>
      <td>0.819406</td>
      <td>1.839765</td>
      <td>1.518348</td>
      <td>2.676672</td>
      <td>0.371961</td>
      <td>0.666341</td>
      <td>0.394153</td>
      <td>0.133679</td>
      <td>0.739214</td>
      <td>-0.044980</td>
    </tr>
    <tr>
      <th>2024-08-18</th>
      <td>0.662743</td>
      <td>0.153703</td>
      <td>0.696028</td>
      <td>-0.207241</td>
      <td>-0.272401</td>
      <td>0.034935</td>
      <td>0.213012</td>
      <td>0.613659</td>
      <td>1.131492</td>
      <td>0.237484</td>
      <td>...</td>
      <td>0.264610</td>
      <td>-0.024573</td>
      <td>-0.562175</td>
      <td>-0.669299</td>
      <td>-0.652221</td>
      <td>0.179133</td>
      <td>-0.370945</td>
      <td>-0.475535</td>
      <td>0.362865</td>
      <td>0.827160</td>
    </tr>
    <tr>
      <th>2024-08-25</th>
      <td>-0.273695</td>
      <td>-0.601815</td>
      <td>-0.448930</td>
      <td>-0.366705</td>
      <td>-0.813017</td>
      <td>-0.853137</td>
      <td>-0.195694</td>
      <td>0.482110</td>
      <td>0.035537</td>
      <td>-0.582116</td>
      <td>...</td>
      <td>-0.238529</td>
      <td>-0.524283</td>
      <td>-0.891178</td>
      <td>-0.745462</td>
      <td>-0.476878</td>
      <td>0.191191</td>
      <td>-0.560147</td>
      <td>-0.239581</td>
      <td>-0.098558</td>
      <td>0.761314</td>
    </tr>
    <tr>
      <th>2024-09-01</th>
      <td>-0.409591</td>
      <td>-0.526109</td>
      <td>-0.567301</td>
      <td>-0.322293</td>
      <td>-0.015178</td>
      <td>-0.927729</td>
      <td>-0.101630</td>
      <td>1.191728</td>
      <td>0.515377</td>
      <td>0.088089</td>
      <td>...</td>
      <td>-0.646062</td>
      <td>-0.806516</td>
      <td>-0.781408</td>
      <td>-0.622814</td>
      <td>-0.403074</td>
      <td>-0.205088</td>
      <td>-0.122978</td>
      <td>-0.700947</td>
      <td>0.281112</td>
      <td>0.589947</td>
    </tr>
    <tr>
      <th>2024-09-08</th>
      <td>0.422580</td>
      <td>0.714202</td>
      <td>0.247782</td>
      <td>1.752364</td>
      <td>-0.149898</td>
      <td>-0.663431</td>
      <td>-0.326403</td>
      <td>2.561017</td>
      <td>1.027955</td>
      <td>0.285407</td>
      <td>...</td>
      <td>1.005430</td>
      <td>-0.196969</td>
      <td>-0.483994</td>
      <td>-0.052976</td>
      <td>-0.189011</td>
      <td>-0.239095</td>
      <td>-0.440169</td>
      <td>-0.707634</td>
      <td>0.361912</td>
      <td>-0.067247</td>
    </tr>
    <tr>
      <th>2024-09-15</th>
      <td>-0.105160</td>
      <td>0.095817</td>
      <td>-0.251665</td>
      <td>0.858608</td>
      <td>-0.449097</td>
      <td>-0.990662</td>
      <td>-0.738098</td>
      <td>0.870204</td>
      <td>0.540200</td>
      <td>-0.318652</td>
      <td>...</td>
      <td>0.742270</td>
      <td>-0.270496</td>
      <td>-0.747327</td>
      <td>-0.531280</td>
      <td>-0.495070</td>
      <td>-0.770200</td>
      <td>-0.522081</td>
      <td>-0.506578</td>
      <td>1.575622</td>
      <td>-0.637280</td>
    </tr>
    <tr>
      <th>2024-09-22</th>
      <td>-0.555423</td>
      <td>-0.612733</td>
      <td>-0.914665</td>
      <td>1.422727</td>
      <td>-0.699540</td>
      <td>-0.759435</td>
      <td>-0.407045</td>
      <td>0.468234</td>
      <td>0.104828</td>
      <td>-0.415065</td>
      <td>...</td>
      <td>0.703289</td>
      <td>-0.772989</td>
      <td>-0.255103</td>
      <td>-0.301772</td>
      <td>-0.309313</td>
      <td>0.103396</td>
      <td>-0.375646</td>
      <td>-0.269931</td>
      <td>0.030363</td>
      <td>0.322973</td>
    </tr>
    <tr>
      <th>2024-09-29</th>
      <td>0.169597</td>
      <td>-0.129038</td>
      <td>-0.689592</td>
      <td>-0.102407</td>
      <td>-0.411760</td>
      <td>-0.682974</td>
      <td>-0.256281</td>
      <td>0.034032</td>
      <td>0.688406</td>
      <td>0.899645</td>
      <td>...</td>
      <td>1.014159</td>
      <td>-0.497108</td>
      <td>-0.698448</td>
      <td>0.768861</td>
      <td>-1.172615</td>
      <td>-0.913951</td>
      <td>-0.930962</td>
      <td>-1.054505</td>
      <td>-0.275822</td>
      <td>-0.181214</td>
    </tr>
    <tr>
      <th>2024-10-06</th>
      <td>-0.741500</td>
      <td>-0.420281</td>
      <td>-0.774669</td>
      <td>-0.272673</td>
      <td>-0.781016</td>
      <td>-1.298667</td>
      <td>-0.750373</td>
      <td>-0.468631</td>
      <td>-0.821706</td>
      <td>-0.916037</td>
      <td>...</td>
      <td>0.510623</td>
      <td>-0.591389</td>
      <td>-1.124977</td>
      <td>-0.782120</td>
      <td>-0.509097</td>
      <td>-0.504648</td>
      <td>-0.548025</td>
      <td>-0.407646</td>
      <td>-0.202848</td>
      <td>-0.702809</td>
    </tr>
    <tr>
      <th>2024-10-13</th>
      <td>-0.217228</td>
      <td>0.051115</td>
      <td>-0.738260</td>
      <td>0.192083</td>
      <td>-0.328425</td>
      <td>-1.076963</td>
      <td>-0.221254</td>
      <td>0.127546</td>
      <td>-0.898737</td>
      <td>-0.476684</td>
      <td>...</td>
      <td>-0.228512</td>
      <td>-0.175055</td>
      <td>-0.738138</td>
      <td>-0.191364</td>
      <td>0.233989</td>
      <td>1.183221</td>
      <td>-0.157005</td>
      <td>-0.036971</td>
      <td>-0.504139</td>
      <td>1.091181</td>
    </tr>
    <tr>
      <th>2024-10-20</th>
      <td>1.104786</td>
      <td>1.142824</td>
      <td>-0.587253</td>
      <td>0.330587</td>
      <td>1.279220</td>
      <td>-0.577072</td>
      <td>0.075697</td>
      <td>0.764391</td>
      <td>0.270819</td>
      <td>-0.071750</td>
      <td>...</td>
      <td>0.350394</td>
      <td>0.097187</td>
      <td>-0.908447</td>
      <td>-0.182454</td>
      <td>-0.155247</td>
      <td>0.310826</td>
      <td>-0.459109</td>
      <td>-0.424622</td>
      <td>-0.477036</td>
      <td>0.433035</td>
    </tr>
    <tr>
      <th>2024-10-27</th>
      <td>-0.030319</td>
      <td>0.472088</td>
      <td>-0.695740</td>
      <td>-0.451541</td>
      <td>-0.200805</td>
      <td>-0.426242</td>
      <td>-0.141880</td>
      <td>-0.370653</td>
      <td>-0.030039</td>
      <td>-0.187639</td>
      <td>...</td>
      <td>-0.086039</td>
      <td>-0.764551</td>
      <td>-0.889248</td>
      <td>-0.234063</td>
      <td>-0.739041</td>
      <td>-0.570494</td>
      <td>-1.038460</td>
      <td>-1.101831</td>
      <td>-0.633060</td>
      <td>0.103803</td>
    </tr>
    <tr>
      <th>2024-11-03</th>
      <td>0.582221</td>
      <td>1.875056</td>
      <td>-0.121092</td>
      <td>0.141083</td>
      <td>-0.072173</td>
      <td>0.234208</td>
      <td>-0.559390</td>
      <td>-0.204361</td>
      <td>-0.628331</td>
      <td>-0.616442</td>
      <td>...</td>
      <td>-0.197272</td>
      <td>-0.120014</td>
      <td>-0.874004</td>
      <td>-0.582806</td>
      <td>0.065851</td>
      <td>-0.151300</td>
      <td>-0.306822</td>
      <td>-0.714731</td>
      <td>0.394331</td>
      <td>0.367188</td>
    </tr>
    <tr>
      <th>2024-11-10</th>
      <td>0.270048</td>
      <td>1.052794</td>
      <td>0.575024</td>
      <td>0.961428</td>
      <td>0.178642</td>
      <td>0.640621</td>
      <td>-0.626758</td>
      <td>-1.072307</td>
      <td>-0.668412</td>
      <td>-0.482881</td>
      <td>...</td>
      <td>-0.071346</td>
      <td>-0.446823</td>
      <td>-0.508321</td>
      <td>-0.960333</td>
      <td>0.005846</td>
      <td>-0.239095</td>
      <td>-0.553710</td>
      <td>-0.248817</td>
      <td>0.090478</td>
      <td>-0.242519</td>
    </tr>
    <tr>
      <th>2024-11-17</th>
      <td>0.507282</td>
      <td>1.521237</td>
      <td>4.250269</td>
      <td>1.630213</td>
      <td>0.352197</td>
      <td>0.037172</td>
      <td>-0.374006</td>
      <td>-0.359959</td>
      <td>0.230098</td>
      <td>0.524283</td>
      <td>...</td>
      <td>0.821202</td>
      <td>-0.336510</td>
      <td>-0.138908</td>
      <td>-0.938368</td>
      <td>-0.384618</td>
      <td>-0.514538</td>
      <td>-0.975960</td>
      <td>-0.813056</td>
      <td>-0.604173</td>
      <td>-0.790604</td>
    </tr>
    <tr>
      <th>2024-11-24</th>
      <td>-0.272939</td>
      <td>0.514381</td>
      <td>0.811916</td>
      <td>0.395862</td>
      <td>-0.293120</td>
      <td>0.094980</td>
      <td>-0.716812</td>
      <td>-0.600125</td>
      <td>1.657620</td>
      <td>0.026676</td>
      <td>...</td>
      <td>1.199453</td>
      <td>-0.530321</td>
      <td>0.306958</td>
      <td>-0.750291</td>
      <td>-0.806338</td>
      <td>-0.448692</td>
      <td>-1.146621</td>
      <td>-0.665463</td>
      <td>-0.714371</td>
      <td>-0.724757</td>
    </tr>
    <tr>
      <th>2024-12-01</th>
      <td>0.060804</td>
      <td>1.328724</td>
      <td>1.041835</td>
      <td>0.255119</td>
      <td>-0.018588</td>
      <td>0.482402</td>
      <td>-0.222271</td>
      <td>0.125007</td>
      <td>0.487926</td>
      <td>-0.216692</td>
      <td>...</td>
      <td>1.036806</td>
      <td>-0.737197</td>
      <td>-0.408212</td>
      <td>-0.741444</td>
      <td>0.091829</td>
      <td>0.179133</td>
      <td>-0.203739</td>
      <td>-0.499891</td>
      <td>-0.652338</td>
      <td>0.301342</td>
    </tr>
    <tr>
      <th>2024-12-08</th>
      <td>-0.402510</td>
      <td>0.126062</td>
      <td>-0.345990</td>
      <td>0.952844</td>
      <td>-0.188046</td>
      <td>-0.418491</td>
      <td>-0.646338</td>
      <td>-0.481366</td>
      <td>-0.262556</td>
      <td>-0.543591</td>
      <td>...</td>
      <td>0.822787</td>
      <td>-0.954402</td>
      <td>-0.821525</td>
      <td>-0.530784</td>
      <td>0.033631</td>
      <td>-0.438801</td>
      <td>-0.713001</td>
      <td>-0.590391</td>
      <td>-0.026317</td>
      <td>0.060223</td>
    </tr>
    <tr>
      <th>2024-12-15</th>
      <td>-0.953107</td>
      <td>-0.089185</td>
      <td>-0.547959</td>
      <td>0.750703</td>
      <td>-0.926800</td>
      <td>0.007955</td>
      <td>-0.676741</td>
      <td>-0.470222</td>
      <td>-0.095777</td>
      <td>-0.354023</td>
      <td>...</td>
      <td>1.737245</td>
      <td>0.225067</td>
      <td>-1.198197</td>
      <td>-0.081039</td>
      <td>-0.588573</td>
      <td>0.013433</td>
      <td>-0.670234</td>
      <td>-0.938145</td>
      <td>0.326870</td>
      <td>0.213547</td>
    </tr>
    <tr>
      <th>2024-12-22</th>
      <td>-0.188513</td>
      <td>0.617501</td>
      <td>-0.116931</td>
      <td>1.846186</td>
      <td>0.448639</td>
      <td>0.881935</td>
      <td>-0.299103</td>
      <td>-0.039263</td>
      <td>0.368131</td>
      <td>0.109173</td>
      <td>...</td>
      <td>2.951443</td>
      <td>-0.263049</td>
      <td>-0.279280</td>
      <td>0.224604</td>
      <td>-0.230015</td>
      <td>0.299968</td>
      <td>-1.122395</td>
      <td>-0.777318</td>
      <td>-0.666760</td>
      <td>-0.768655</td>
    </tr>
    <tr>
      <th>2024-12-29</th>
      <td>-0.546487</td>
      <td>-0.060161</td>
      <td>-0.880570</td>
      <td>-0.644669</td>
      <td>-1.031624</td>
      <td>-0.194965</td>
      <td>-0.860776</td>
      <td>-0.516733</td>
      <td>-1.406117</td>
      <td>-1.289490</td>
      <td>...</td>
      <td>0.578743</td>
      <td>-1.154338</td>
      <td>-0.959794</td>
      <td>-1.342783</td>
      <td>-0.468570</td>
      <td>-0.165291</td>
      <td>-0.907888</td>
      <td>-1.152378</td>
      <td>-0.962603</td>
      <td>-1.272842</td>
    </tr>
  </tbody>
</table>
<p>52 rows × 25 columns</p>
</div>




```python
merged_df.to_excel('merged_df.xlsx')
```


```python

```
