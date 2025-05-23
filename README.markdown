# Snoog Forex Trading Suite

A web-based forex trading dashboard built with FastAPI (backend) and React (frontend), integrating with MetaTrader 5 for real-time signals, charts, and trade execution.

## Project Structure
- `backend/`: FastAPI backend with API and WebSocket endpoints.
- `frontend/`: React frontend with Tailwind CSS and GSAP animations.
- `utils/`: Existing utilities for database, MT5 trading, forecasting, plotting, and configuration.

## Setup
1. **Clone the Repository**:
   ```bash
   git clone <your-repo-url>
   cd trading-app
   ```

2. **Backend Setup**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

3. **Frontend Setup**:
   ```bash
   cd frontend
   npm install
   npm start
   ```

4. **Environment Variables**:
   Create a `.env` file in the `backend` directory:
   ```env
   MT5_LOGIN=your_mt5_login
   MT5_PASSWORD=your_mt5_password
   MT5_SERVER=your_mt5_server
   MYSQL_HOST=your_mysql_host
   MYSQL_USER=your_mysql_user
   MYSQL_PASSWORD=your_mysql_password
   MYSQL_DATABASE=trading_insights
   MYSQL_PORT=3306
   ```

## Deployment
1. Push to GitHub.
2. Deploy `backend` and `frontend` to Vercel separately.
3. Set environment variables in Vercel for the backend.
4. Set `REACT_APP_API_URL` in the frontend Vercel project to the backend URL.

## Usage
- Access the dashboard at the frontend URL.
- Select symbol and timeframe to view signals, charts, and metrics.
- Execute trades manually or configure auto-trading.
- Download historical data and run forecasts.

## License
MIT