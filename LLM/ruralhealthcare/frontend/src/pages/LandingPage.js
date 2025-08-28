import React from 'react';
import { Box, Button, Typography, Paper } from '@mui/material';
import { useNavigate } from 'react-router-dom';

const LandingPage = () => {
  const navigate = useNavigate();

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #e3f0ff 0%, #f9f9f9 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <Paper
        elevation={6}
        sx={{
          p: 6,
          borderRadius: 4,
          minWidth: 350,
          textAlign: 'center',
          boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.15)',
        }}
      >
        <Typography variant="h2" sx={{ fontWeight: 700, mb: 2, color: '#1976d2' }}>
          Medi-samvaad
        </Typography>
        <Typography variant="h6" sx={{ mb: 4, color: '#333' }}>
          Your Health, Your Language
        </Typography>
        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
          <Button
            variant="contained"
            color="primary"
            size="large"
            sx={{ px: 4, fontWeight: 600 }}
            onClick={() => navigate('/login')}
          >
            LOGIN
          </Button>
          <Button
            variant="outlined"
            color="primary"
            size="large"
            sx={{ px: 4, fontWeight: 600 }}
            onClick={() => navigate('/register')}
          >
            REGISTER
          </Button>
        </Box>
      </Paper>
    </Box>
  );
};

export default LandingPage;
