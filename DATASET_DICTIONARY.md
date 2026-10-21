# Dataset Schema, Data Dictionary & Geospatial Reference
## AI-Based Business Location Selector Using Machine Learning

- **Dataset File:** `business_location_synthetic_dataset.csv`
- **Total Records:** 1,001 rows
- **Total Attributes:** 18 columns
- **Geography:** Bhopal, Madhya Pradesh, India (Urban Catchment Hubs)

---

## 1. Complete Field Dictionary

| Column Name | Data Type | Permissible Range / Values | Unit / Scale | Description & Business Domain Significance |
| :--- | :---: | :---: | :---: | :--- |
| `location_name` | String / Object | 18 Bhopal Localities | Categorical | Specific urban or suburban candidate locality being evaluated. |
| `business_type` | String / Object | `Cafe`, `Pharmacy`, `Grocery Store`, `Clothing Shop` | Categorical | The retail or service category under evaluation. Affects feature relevance and weight profiles. |
| `monthly_rent` | Integer | 12,000 – 85,000 | INR (₹) / month | Anticipated monthly lease cost. Direct operational expense impacting margin viability. |
| `population` | Integer | 7,000 – 60,000 | Persons | Catchment demographic size residing within 2–3 km travel radius. |
| `competitor_count` | Integer | 1 – 15 | Count | Number of directly competing businesses operating within 1 km radius. |
| `daily_footfall` | Integer | 200 – 3,000 | Pedestrians / day | Estimated volume of pedestrians and walk-in passersby fronting the candidate site. |
| `accessibility_score` | Integer | 1 – 10 | Rating (1-10) | Road quality, sidewalk continuity, and vehicular ingress/egress ease. |
| `parking_score` | Integer | 1 – 10 | Rating (1-10) | Availability and convenience of customer two-wheeler and four-wheeler parking slots. |
| `public_transport_score` | Integer | 1 – 10 | Rating (1-10) | Proximity and frequency of public bus stops, BRTS stations, and auto-rickshaw stands. |
| `income_level` | String / Object | `Low`, `Medium`, `High` | Ordinal | Purchasing power of the local catchment population. Guides pricing strategy. |
| `nearby_facilities` | String / Object | `Offices`, `Market`, `College`, `Hospital`, `Residential`, `Mixed` | Categorical | Primary anchor institution generating customer gravity and footfall traffic. |
| `target_demand` | Float / Int | 30 – 220 | Daily Customers | Primary continuous target variable used for supervised regression model training. |
| `demand_score` | Float | 0.0 – 100.0 | Index [0–100] | Min-Max normalized score representing relative commercial demand strength. |
| `footfall_score` | Float | 0.0 – 100.0 | Index [0–100] | Min-Max normalized representation of pedestrian footfall. |
| `mobility_score` | Float | 0.0 – 100.0 | Index [0–100] | Consolidated index combining accessibility, parking, and transit scores. |
| `competition_score` | Float | 0.0 – 100.0 | Index [0–100] | Opportunity index reflecting competitive saturation (higher score = lower saturation / higher opportunity). |
| `rent_affordability_score`| Float | 0.0 – 100.0 | Index [0–100] | Relative score representing financial rent viability. |
| `business_potential` | String / Object | `Good`, `Moderate`, `Low` | Categorical | Discrete classification label indicating overall commercial viability tier. |

---

## 2. Bhopal Candidate Localities & Geospatial Coordinate Lookup

For interactive Folium mapping and spatial analytics, the table below documents the geographic centroid coordinates for all localities represented in the dataset:

| Locality Name | Latitude | Longitude | Zone / Character |
| :--- | :---: | :---: | :--- |
| **New Market** | `23.2332` | `77.4013` | Central Commercial Core (High footfall, retail hub) |
| **MP Nagar (Zone I & II)** | `23.2324` | `77.4338` | Commercial & Corporate Business District |
| **Arera Colony** | `23.2120` | `77.4277` | Premium Affluent Residential & Upmarket Cafes |
| **Hoshangabad Road** | `23.1950` | `77.4470` | Commercial Highway Corridor, Malls & Showrooms |
| **Kolar Road** | `23.1780` | `77.4190` | Rapidly Expanding Residential & Retail Strip |
| **Lalghati** | `23.2840` | `77.3780` | Transit Junction connecting airport and Old Bhopal |
| **Indrapuri** | `23.2500` | `77.4660` | Student & Residential Hub near BHEL/Engineering colleges |
| **BHEL Township** | `23.2610` | `77.4770` | Industrial & Institutional Township with steady demand |
| **Bawadia Kalan** | `23.1930` | `77.4360` | Modern Residential Enclave, High Purchasing Power |
| **Saket Nagar** | `23.2200` | `77.4520` | Residential Hub adjacent to AIIMS Bhopal |
| **Gulmohar Colony** | `23.2040` | `77.4300` | High-Income Residential, Boutique Cafes & Salons |
| **Karond** | `23.3030` | `77.3980` | Growing Suburb with Wholesale Mandi & Transport Hub |
| **TT Nagar** | `23.2380` | `77.4040` | Established City Center, Administrative & Sports Hub |
| **Berasia Road** | `23.2750` | `77.4050` | High Density Wholesale & Value Retail Area |
| **Govindpura** | `23.2530` | `77.4480` | Industrial Area with High Workforce Inflow |
| **Misrod** | `23.1670` | `77.4650` | Southern Growth Corridor, Auto Showrooms & Tech Parks |
| **Shahpura** | `23.2010` | `77.4210` | Scenic Lake-Facing Commercial and Dining District |
| **Habibganj (Rani Kamlapati)** | `23.2167` | `77.4410` | Modernized Transit Hub, Hotels & High-Footfall Plaza |

---

## 3. Data Integrity & Validation Rules

1. **Range Validation:**
   - `monthly_rent` must satisfy: $10,000 \le \text{rent} \le 150,000$.
   - `population` must satisfy: $5,000 \le \text{population} \le 100,000$.
   - `daily_footfall` must satisfy: $100 \le \text{footfall} \le 10,000$.
   - `competitor_count` must satisfy: $0 \le \text{count} \le 30$.
   - All 1–10 scores must fall strictly in $[1, 10]$.
2. **Categorical Consistency:**
   - `business_type` must match one of the four supported categories: `Cafe`, `Pharmacy`, `Grocery Store`, `Clothing Shop`.
   - `income_level` must be one of: `Low`, `Medium`, `High`.
   - `business_potential` must be one of: `Good`, `Moderate`, `Low`.
3. **Synthetic Nature Disclosure:**
   - Generated values reflect realistic commercial distributions and correlations characteristic of Bhopal's urban retail economy.
   - Designed to demonstrate data science workflows, model benchmarking, and geospatial dashboards in an academic setting.
