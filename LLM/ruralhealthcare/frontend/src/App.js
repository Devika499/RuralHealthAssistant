import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { LocalizationProvider } from '@mui/x-date-pickers';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import Layout from './layouts/Layout';

// Pages
import Login from './pages/Login';
import Register from './pages/Register';
import Chat from './pages/Chat';
import Profile from './pages/Profile';
import MedicalHistoryPage from './pages/MedicalHistory';
import LandingPage from './pages/LandingPage';
import Medication from './pages/Medical';
import SymptomChecker from './pages/SymptomChecker'; // <-- Add this
import MedicalTermSimplifier from './pages/MedicalTermSimplifier';
import DietRecommendation from './pages/DietRecommendation';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <LocalizationProvider dateAdapter={AdapterDateFns}>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route element={<Layout />}>
            <Route path="/chat" element={<Chat />} />
            <Route path="/profile" element={<Profile />} />
            <Route path="/medical-history" element={<MedicalHistoryPage />} />
            <Route path="/medications" element={<Medication />} />
            <Route path="/symptom-checker" element={<SymptomChecker />} />
            <Route path="/simplify-term" element={<MedicalTermSimplifier />} />
            <Route path="/diet-recommendation" element={<DietRecommendation />} />
          </Route>
        </Routes>
      </LocalizationProvider>
    </ThemeProvider>
  );
}

export default App;