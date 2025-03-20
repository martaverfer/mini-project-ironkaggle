# EDA

Dataset has shape (21613, 21)

## Numerical columns
Dataset has numerical data in columns: 
- Descrete: `['price', 'view', 'bedrooms', 'bathrooms', 'floors', 'zipcode', 'yr_built', 'yr_renovated']`
- Continuous: `['price', 'sqft_lot', 'sqft_lot15', 'sqft_above', 'sqft_basement', 'sqft_living', 'lat', 'long']`

### Descrete values

- Column `"view"` has 5 unique values.
  - Unique values are: `[0 3 4 2 1]`
   - Number of times the house has been viewed.

- Column `"yr_built"` has 116 unique values.
  - The year the house was built.
- Column `"yr_renovated"` has 70 unique values.
  - The year the house was renovated.

#### Countable features of the houses

- Column `"bathrooms"` has 30 unique values.
  - Number of bathrooms in the house, per bedroom.

- Column `"bedrooms"` has 13 unique values.
  - Number of bedrooms in the house.
  - Unique values are:
 `[ 3  2  4  5  1  6  7  0  8  9 11 10 33]`

- Column `"floors"` has 6 unique values.
  -  Number of floors (levels) in the house.
  - Unique values are:
 `[1.  2.  1.5 3.  2.5 3.5]`

### Contiuous values
- Column `"price"` has 4028 unique values.
  - The sale price of the house (prediction target).
  - Dependent variable, the target.
  - Our primary focus is to understand which features most significantly impact the house price. Additionally, we aim to explore properties valued at $650K and above for more detailed insights.


#### Size related features of the houses
- Column `"sqft_lot"` has 9782 unique values.
  - Square footage of the land space.

- Column `"sqft_lot15"` has 8689 unique values.
  - The land spaces for the nearest 15 neighbors in 2015.

- Column `"sqft_living"` has 1038 unique values.
  - Square footage of the interior living space.

- Column `"sqft_above"` has 946 unique values.
  - Square footage of the house apart from the basement.
  
- Column `"sqft_living15"` has 777 unique values.
  - The interior living space for the nearest 15 neighbors in 2015.
  
- Column `"sqft_basement"` has 306 unique values.
  - Square footage of the basement.
  
For all columns starting with prefix `"sqft_"`:
  - All values are integers, but from scatter plot distribution looks like the values can be treated as continuous.
  - Values are highly askewed.

#### Geo location related features of the houses
- Column `"lat"` has 5034 unique values.
  - lat: Latitude coordinate.
  - Continuous.

- Column `"long"` has 752 unique values.
  - long: Longitude coordinate.
  - Continuous.

- Column `"zipcode"` has 70 unique values.
  - ZIP code area.

## Categorical data

Dataset has categorical data in columns:
```['date', 'id', 'waterfront', 'grade', 'condition']```

- Column `"date"` has 372 unique values.
  - The date on which the house was sold.
  - `[IDEA]`: convert to timestamp?

- Column `"id"` has 21436 unique values.
  - `"id"` column identifies house for sale. 
  - 177 IDs are repeated, that means some houses were listed for sale several times. It is less than 1%.
  - What changes in price and other features for repeated sales?
  - Not a feature for models.

- Column `"waterfront"` has 2 unique values.
  - Whether the house has a waterfront view.
  - Unique values are: `[0 1]`
  - Whether the house has a waterfront view.
  - Boolean value.

- Column `"grade"` has 12 unique values.
  - The overall grade given to the house, based on the King County grading system.
  - Unique values are:
 `[ 7  6  8 11  9  5 10 12  4  3 13  1]`
  - The overall grade given to the house, based on the King County grading system.

- Column `"condition"` has 5 unique values.
  - The overall condition of the house.
  - Unique values are: `[3 5 4 1 2]`
  - The overall condition of the house.
  - Ordinal data.