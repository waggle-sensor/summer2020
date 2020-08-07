# Nowcasting over the Edge 
## by Rick Nueve

---

### What is Nowcasting?
<p>
Nowcasting is the task of weather forecasting within a horizon period of twenty-four hours.<sup>2</sup> By
being able to accurately nowcast, abrupt weather changes can be announced earlier. This could allow
people to have the needed time to prepare for weather-related dangers such as hail, thunder, or tornados.
</p>

---

### Nowcasting and Deep Learning
<p>
Traditionally, nowcasting is performed by Numerical Weather Prediction (NWP) models that use radar data. However, 
recently, many scientists have developed ways to conduct nowcasting using Deep Learning and have achieved great success.
As of March 2020, Google Research released a paper documenting a Deep Learning model that performs nowcasting called MetNet.<sup>1</sup>
MetNet was able to outperform the High-Resolution Rapid Refresh (HRRR) system, the state of the art NWP method available from
NOAA (the National Oceanic and Atmospheric Administration) for precipitation forecasting within a 7-8 hour window. Both NWP models and 
Deep Learning nowcasting models use radar data as input. Radar data does provide powerful insight into macro
atmospheric activity. However, I hypothesize that adding data from ground-based sensors to nowcasting models could improve nowcasting
for precise locations. To be able to accurately nowcast for a precise location, it is reasonable to assume that information about the site is
needed- which is not provided through traditional radar data. 
</p>
<p>
To explore the plausibility of using ground-based sensors to enhance the performance of nowcasting models,
I sought out to develop a rudimentary experiment that would provide insight into my hypothesis. I desired to construct a Deep Learning model that only used ground-based sensor data to perform nowcasting. The reasoning was that if I was able to perform nowcasting using only ground-based sensor data (which is unique from radar data), then it would be plausible that combining ground-based sensor data with radar data could create a more powerful model. Vice-versa, if a Deep Learning model that only used ground-based sensor data could not perform nowcasting to any degree, it would be unlogical to assume that combining ground-based sensor data and radar data would improve a model's ability to nowcast- thus voiding the plausibility of my hypothesis. 
</p>

---

### Why Sage?

---

### WeatherNet?

---

### Conclusion

---

### About the Author

---

## Refrence
[1] Casper Kaae Sønderby et al.MetNet: A Neural Weather  Model  for  Precipitation  Forecasting.2020. arXiv:2003.12140 [cs.LG]. <br>
[2] Yong  Wang  et  al.Guidelines  for  Nowcasting Techniques. Nov. 2017.ISBN: 978-92-63-11198-2 <br>
