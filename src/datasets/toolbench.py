import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path


class ToolBenchDataset:
    def __init__(self, n_tools: int = 500, seed: int = 42):
        np.random.seed(seed)
        self.n_tools = n_tools
        
        self.tools = self._generate_synthetic_tools()
        self.queries = self._generate_synthetic_queries()
    
    def _generate_synthetic_tools(self) -> List[Dict]:
        categories = [
            "Travel", "Finance", "Weather", "Food", "Entertainment",
            "Shopping", "Communication", "Health", "Education", "Productivity",
            "Social", "News", "Sports", "Maps", "Utilities"
        ]
        
        tool_templates = {
            "Travel": ["SearchFlights", "BookHotel", "GetCarRental", "CheckTrainSchedule", "FindTours"],
            "Finance": ["GetStockPrice", "ConvertCurrency", "CheckBankBalance", "TransferMoney", "GetCryptoPrice"],
            "Weather": ["GetCurrentWeather", "GetForecast", "CheckAirQuality", "GetSunriseSunset", "GetUVIndex"],
            "Food": ["FindRestaurants", "OrderFood", "GetRecipes", "CheckFoodDelivery", "ReserveTable"],
            "Entertainment": ["SearchMovies", "GetMusicRecommendations", "FindEvents", "StreamVideo", "GetGameInfo"],
            "Shopping": ["SearchProducts", "ComparePrice", "TrackPackage", "FindDeals", "CheckInventory"],
            "Communication": ["SendEmail", "SendSMS", "MakeCall", "ScheduleMeeting", "TranslateText"],
            "Health": ["FindDoctors", "CheckSymptoms", "GetMedicationInfo", "TrackFitness", "BookAppointment"],
            "Education": ["SearchCourses", "GetDefinition", "SolveMath", "TranslateLanguage", "FindTutors"],
            "Productivity": ["CreateDocument", "ManageTasks", "SetReminder", "ConvertFile", "ScanDocument"],
            "Social": ["PostUpdate", "SearchProfiles", "GetTrending", "AnalyzeSentiment", "FindInfluencers"],
            "News": ["GetHeadlines", "SearchArticles", "GetLocalNews", "FactCheck", "GetWeeklyDigest"],
            "Sports": ["GetScores", "GetStandings", "GetPlayerStats", "FindGames", "GetOdds"],
            "Maps": ["GetDirections", "FindNearby", "CalculateDistance", "GetTraffic", "SearchPlaces"],
            "Utilities": ["GenerateQRCode", "ShortenURL", "CalculateTax", "ConvertUnits", "GeneratePassword"]
        }
        
        tools = []
        tool_id = 0
        
        for category, tool_names in tool_templates.items():
            for tool_name in tool_names:
                if tool_id >= self.n_tools:
                    break
                
                description = self._generate_tool_description(tool_name, category)
                parameters = self._generate_tool_parameters(tool_name)
                
                tools.append({
                    "tool_id": tool_id,
                    "name": tool_name,
                    "category": category,
                    "description": description,
                    "parameters": parameters,
                    "success_rate": np.random.uniform(0.7, 0.95)
                })
                tool_id += 1
            
            if tool_id >= self.n_tools:
                break
        
        while tool_id < self.n_tools:
            category = np.random.choice(categories)
            tool_name = f"Custom{category}Tool{tool_id}"
            description = f"A custom tool for {category.lower()} related tasks. Provides specialized functionality."
            
            tools.append({
                "tool_id": tool_id,
                "name": tool_name,
                "category": category,
                "description": description,
                "parameters": {"input": "string"},
                "success_rate": np.random.uniform(0.6, 0.9)
            })
            tool_id += 1
        
        return tools
    
    def _generate_tool_description(self, tool_name: str, category: str) -> str:
        descriptions = {
            "SearchFlights": "Search for available flights between two locations with flexible date options and fare comparison.",
            "BookHotel": "Book hotel accommodations with options for room type, dates, and special requests.",
            "GetCurrentWeather": "Get the current weather conditions including temperature, humidity, and precipitation.",
            "FindRestaurants": "Search for restaurants based on cuisine, location, rating, and price range.",
            "GetStockPrice": "Retrieve current stock price and trading information for a given ticker symbol.",
            "SendEmail": "Send an email to specified recipients with subject and body content.",
            "SearchProducts": "Search for products across online retailers with filtering options.",
            "GetDirections": "Get navigation directions between two locations with route options.",
        }
        
        return descriptions.get(tool_name, 
            f"A {category.lower()} tool that performs {tool_name} operations. "
            f"Useful for {category.lower()}-related queries and tasks.")
    
    def _generate_tool_parameters(self, tool_name: str) -> Dict:
        common_params = {
            "SearchFlights": {"origin": "string", "destination": "string", "date": "string"},
            "BookHotel": {"location": "string", "check_in": "string", "check_out": "string"},
            "GetCurrentWeather": {"location": "string"},
            "FindRestaurants": {"location": "string", "cuisine": "string"},
            "GetStockPrice": {"ticker": "string"},
            "SendEmail": {"to": "string", "subject": "string", "body": "string"},
            "SearchProducts": {"query": "string", "category": "string"},
            "GetDirections": {"origin": "string", "destination": "string"},
        }
        
        return common_params.get(tool_name, {"input": "string"})
    
    def _generate_synthetic_queries(self) -> Dict[str, List[Dict]]:
        queries = {
            "I1": [],
            "I2": [],
            "I3": []
        }
        
        for _ in range(100):
            category = np.random.choice([t["category"] for t in self.tools])
            matching_tools = [t for t in self.tools if t["category"] == category]
            if matching_tools:
                tool = np.random.choice(matching_tools)
                queries["I1"].append({
                    "query": f"I need to {tool['name'].lower().replace('get', 'find ').replace('search', 'look for ')}",
                    "ground_truth": [tool["tool_id"]],
                    "category": category
                })
        
        categories = list(set(t["category"] for t in self.tools))
        for _ in range(50):
            category = np.random.choice(categories)
            matching_tools = [t for t in self.tools if t["category"] == category]
            if len(matching_tools) >= 2:
                selected = np.random.choice(matching_tools, size=min(3, len(matching_tools)), replace=False)
                queries["I2"].append({
                    "query": f"Help me with multiple {category.lower()} tasks",
                    "ground_truth": [t["tool_id"] for t in selected],
                    "category": category
                })
        
        for _ in range(50):
            selected_categories = np.random.choice(categories, size=min(3, len(categories)), replace=False)
            selected_tools = []
            for cat in selected_categories:
                matching = [t for t in self.tools if t["category"] == cat]
                if matching:
                    selected_tools.append(np.random.choice(matching))
            
            if len(selected_tools) >= 2:
                queries["I3"].append({
                    "query": f"I need help with {', '.join(selected_categories[:2])} and more",
                    "ground_truth": [t["tool_id"] for t in selected_tools],
                    "categories": list(selected_categories)
                })
        
        return queries
    
    def get_tool_texts(self) -> List[str]:
        return [f"{tool['name']}: {tool['description']}" for tool in self.tools]
    
    def get_tool_by_id(self, tool_id: int) -> Dict:
        return self.tools[tool_id]
    
    def execute_tool(self, tool_id: int, query_context: Optional[Dict] = None) -> Tuple[Dict, bool]:
        tool = self.tools[tool_id]
        success = np.random.random() < tool["success_rate"]
        
        result = {
            "tool_id": tool_id,
            "tool_name": tool["name"],
            "success": success,
            "output": f"Result from {tool['name']}" if success else "Tool execution failed"
        }
        
        return result, success
    
    def get_queries(self, task_type: str = "I1") -> List[Dict]:
        return self.queries.get(task_type, [])
    
    def __len__(self) -> int:
        return self.n_tools
