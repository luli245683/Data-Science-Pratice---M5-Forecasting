
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="img/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">M5-Forecasting Competition</h3>
  <h5>The challenge of estimating the unit sales of Walmart retail goods</h5>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)




## About The Project
- Estimate the point forecasts of the unit sales of various products sold in the USA by Walmart.

- You will use hierarchical sales data from Walmart, the world’s largest company by revenue, to forecast daily sales for the next 28 days.

-  The data, covers stores in three US States (California, Texas, and Wisconsin) and includes **item level, department, product categories, and store details**. In addition, it has explanatory variables such as price, promotions, day of the week, and special events. Together, this robust dataset can be used to improve forecasting accuracy.

- If successful, the methods used can be applied in various business areas such as **setting up appropriate inventory** or **service levels**.
### Column Explanation

#### Calendar

- Contains information about the dates on which the products are sold.

-   |Column Name |Description|
    |-----|--------|
    |date|The date in a “y-m-d” format.      |
    |wm_yr_wk| The id of the week the date belongs to.|
    |weekday| The type of the day (Saturday, Sunday, …, Friday).|
    |wday| The id of the weekday, starting from Saturday.|
    |month| The month of the date.|
    |year| The year of the date.| 
    |event_name_1| If the date includes an event, the name of this event such as **SuperBowl**|
    |event_type_1| If the date includes an event, the type of this event such as **Sporting**.|
    |event_name_2| If the date includes a second event, the name of this event.|
    |event_type_2| If the date includes a second event, the type of this event.|
    |snap_CA, snap_TX, and snap_WI|A binary variable (0 or 1) indicating whether the stores of CA, TX or WI allow SNAP  purchases on the examined date. 1 indicates that SNAP purchases are allowed.|

#### Sales train

-  Contains the historical daily unit sales data per product and store
-   |Column Name |Description|
    |-----|--------|
    |id|IDs       |
    |item_id  |IDs of item      |
    |dept_id  |IDs of department      |
    |cat_id  |IDs of category      |
    |store_id  |IDs of store       |
    |state_id  |IDs of state      |
    |d_x  |The number of units sold at day i, starting from 2011-01-29    
    
#### Sell prices

-  Contains information about the price of the products sold per store and date.

-   |Column Name | Description|
    |-----|---------|
    |store_id| The id of the store where the product is sold.| 
    |item_id| The id of the product.|
    |wm_yr_wk| The id of the week.|
    |sell_price| The price of the product for the given week/store. The price is provided **per week (average across seven days)**. If not available, this means that the product was not sold during the examined week. Note that although prices are constant at weekly basis, they may change through time (both training and test set).| 
    
### Built With
- OS: Windows 10 
- Python version: 3.6



<!-- GETTING STARTED -->
## Getting Started
- The main file is under the scr folder.
### Prerequisites
- Please make sure you have installed all the requirement libraries which in the requirements.txt .

