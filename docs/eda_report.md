# EDA

Dataset has shape (21613, 21)

## Numerical columns
Dataset has numerical data in columns: 


### Discrete values

- Column "view" has 5 unique values.
   -- Unique values are:
 [0 3 4 2 1]
   - Number of times the house has been viewed.

- Column "yr_built" has 116 unique values.
  - [IDEA]: create new column with age of the building from this.
- Column "yr_renovated" has 70 unique values.
  - [IDEA]: create new column with years since last renovation.

#### Countable features

- Column "bathrooms" has 30 unique values.
- Column "bedrooms" has 13 unique values.
   -- Unique values are:
 [ 3  2  4  5  1  6  7  0  8  9 11 10 33]
- Column "floors" has 6 unique values.
   -- Unique values are:
 [1.  2.  1.5 3.  2.5 3.5]

### Continuous values
- Column "price" has 4028 unique values.
  - Dependent variable, the target.
  - Our primary focus is to understand which features most significantly impact the house price. Additionally, we aim to explore properties valued at $650K and above for more detailed insights.


#### Size related
- Column "sqft_lot" has 9782 unique values.
  - sqft_lot: Square footage of the land space.
- Column "sqft_lot15" has 8689 unique values.
  - sqft_lot15: The land spaces for the nearest 15 neighbors in 2015.
- Column "sqft_living" has 1038 unique values.
  - sqft_living: Square footage of the interior living space.
  - TODO: copy from dataset desceription https://my.ironhack.com/cohorts/66f469cb0c31b7002bd4e62b/lms/courses/course-v1:IRONHACK+DSMLFT+202502_RMT/modules/ironhack-course-chapter_6/units/ironhack-course-chapter_6-sequential_4-vertical
- Column "sqft_above" has 946 unique values.
- Column "sqft_living15" has 777 unique values.
- Column "sqft_basement" has 306 unique values.
  
For all columns starting with prefix "sqft_":
  - All integers, but from scatter plot distribution looks like the values can be treated as continuous.
 



#### Geo location related
- Column "lat" has 5034 unique values.
- Column "long" has 752 unique values.
  - lat: Latitude coordinate.
  - long: Longitude coordinate.
  - Continuous.
- Column "zipcode" has 70 unique values.
  - TODO



## Categorical data

- Column "date" has 372 unique values.

- Column "id" has 21436 unique values.
  - "id" column identifies house for sale. 
  - 177 IDs are repeated, that means some houses were listed for sale several times. It is less than 1%.
  - What changes in price and other features for repeated sales?
  - Not a feature for models.

- Column "waterfront" has 2 unique values.
   -- Unique values are:
 [0 1]
  - Whether the house has a waterfront view.
  - Boolean value.
  - [Question]: keep it as 0/1 or True/False? 
    -> Easier for calculations to keep as digits, but for plots to be mapped to True/False

- Column "grade" has 12 unique values.
   -- Unique values are:
 [ 7  6  8 11  9  5 10 12  4  3 13  1]
  - The overall grade given to the house, based on the King County grading system.

- Column "condition" has 5 unique values.
  -- Unique values are:
  [3 5 4 1 2]
  - The overall condition of the house.
  - Ordinal data.