export interface WeatherInfo {
  temp: string;
  condition: string;
}

const normalizeCondition = (code?: number, fallback = 'Cloudy') => {
  if (!code) return fallback;
  if ([113].includes(code)) return 'Sunny';
  if ([116, 119, 122, 143, 248, 260].includes(code)) return 'Cloudy';
  if ([176, 263, 266, 293, 296, 299, 302, 305, 308, 353, 356, 359].includes(code)) return 'Rainy';
  if ([200, 386, 389, 392, 395].includes(code)) return 'Stormy';
  if ([179, 182, 185, 227, 230, 281, 284, 311, 314, 317, 320, 323, 326, 329, 332, 335, 338, 350, 362, 365, 368, 371, 374, 377].includes(code)) return 'Rainy';
  return fallback;
};

export const getCurrentWeather = async (city: string): Promise<WeatherInfo> => {
  const response = await fetch(`https://wttr.in/${encodeURIComponent(city)}?format=j1`);
  if (!response.ok) {
    throw new Error(`Weather request failed: ${response.status}`);
  }

  const data = await response.json();
  const current = data?.current_condition?.[0];
  if (!current?.temp_C) {
    throw new Error('Weather response did not include current temperature');
  }

  return {
    temp: String(Math.round(Number(current.temp_C))),
    condition: normalizeCondition(
      Number(current.weatherCode),
      current.weatherDesc?.[0]?.value || 'Cloudy'
    ),
  };
};
