import React, { useState } from 'react';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  Alert,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';

const Register = () => {
  const navigate = useNavigate();
  const { register } = useAuth();
  const [formData, setFormData] = useState({
    mobileNumber: '',
    password: '',
    confirmPassword: '',
    name: '',
    birthdate: '',
    preferredLanguage: 'hi',
  });
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    try {
      await register(formData);
      setSuccess('Registration successful');
      // Redirect to login after a short delay to show success message
      setTimeout(() => {
        navigate('/login');
      }, 1000);
    } catch (err) {
      console.error('Registration error:', err);
      setError(err.message || 'Registration failed');
    }
  };

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
          minWidth: 400,
          textAlign: 'center',
          boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.15)',
        }}
      >
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 3, color: '#1976d2' }}>
          Register
        </Typography>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        {success && (
          <Alert severity="success" sx={{ mb: 2 }}>
            {success}
          </Alert>
        )}
        <form onSubmit={handleSubmit}>
          <TextField
            fullWidth
            label="Mobile Number"
            name="mobileNumber"
            value={formData.mobileNumber}
            onChange={handleChange}
            margin="normal"
            required
          />

          <TextField
            fullWidth
            label="Name"
            name="name"
            value={formData.name}
            onChange={handleChange}
            margin="normal"
            required
          />

          <TextField
            fullWidth
            label="Birthdate"
            type="date"
            name="birthdate"
            value={formData.birthdate}
            onChange={handleChange}
            margin="normal"
            required
            InputLabelProps={{
              shrink: true,
            }}
          />

          <TextField
            fullWidth
            label="Password"
            name="password"
            type="password"
            value={formData.password}
            onChange={handleChange}
            margin="normal"
            required
          />

          <TextField
            fullWidth
            label="Confirm Password"
            name="confirmPassword"
            type="password"
            value={formData.confirmPassword}
            onChange={handleChange}
            margin="normal"
            required
          />

          <Button
            type="submit"
            fullWidth
            variant="contained"
            color="primary"
            sx={{ fontWeight: 600, mb: 2 }}
          >
            REGISTER
          </Button>
        </form>

        <Typography variant="body2" sx={{ mt: 2 }}>
          ALREADY HAVE AN ACCOUNT?{' '}
          <Button variant="text" onClick={() => navigate('/login')} sx={{ color: '#1976d2', fontWeight: 600 }}>
            LOGIN
          </Button>
        </Typography>
      </Paper>
    </Box>
  );
};

export default Register;
