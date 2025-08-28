import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { useAuth } from '../hooks/useAuth';

const AuthForm = ({ type }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const { login, register } = useAuth();
  const [formData, setFormData] = useState({
    mobileNumber: '',
    password: '',
    name: '',
    birthdate: '',
    preferredLanguage: 'hi',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const languages = [
    { value: 'hi', label: 'Hindi' },
    { value: 'en', label: 'English' },
    { value: 'bn', label: 'Bengali' },
    { value: 'te', label: 'Telugu' },
    { value: 'ta', label: 'Tamil' },
  ];

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      if (type === 'login') {
        await login(formData.mobileNumber, formData.password);
      } else {
        await register({
          ...formData,
          birthdate: new Date(formData.birthdate),
        });
      }
      navigate('/chat', { replace: true });
    } catch (err) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container component="main" maxWidth="xs">
      <Box
        sx={{
          marginTop: 8,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        <Paper elevation={3} sx={{ p: 4, width: '100%' }}>
          <Typography component="h1" variant="h5">
            {type === 'login' ? 'Login' : 'Register'}
          </Typography>
          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
          <Box component="form" onSubmit={handleSubmit} sx={{ mt: 1 }}>
            <TextField
              margin="normal"
              required
              fullWidth
              id="mobileNumber"
              label="Mobile Number"
              name="mobileNumber"
              autoComplete="mobile"
              autoFocus
              value={formData.mobileNumber}
              onChange={handleChange}
              disabled={loading}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              name="password"
              label="Password"
              type="password"
              id="password"
              autoComplete="current-password"
              value={formData.password}
              onChange={handleChange}
              disabled={loading}
            />
            {type === 'register' && (
              <>
                <TextField
                  margin="normal"
                  required
                  fullWidth
                  name="name"
                  label="Full Name"
                  id="name"
                  value={formData.name}
                  onChange={handleChange}
                  disabled={loading}
                />
                <TextField
                  margin="normal"
                  required
                  fullWidth
                  name="birthdate"
                  label="Birthdate"
                  type="date"
                  id="birthdate"
                  InputLabelProps={{
                    shrink: true,
                  }}
                  value={formData.birthdate}
                  onChange={handleChange}
                  disabled={loading}
                />
                <FormControl fullWidth margin="normal">
                  <InputLabel>Preferred Language</InputLabel>
                  <Select
                    name="preferredLanguage"
                    value={formData.preferredLanguage}
                    onChange={handleChange}
                    label="Preferred Language"
                    disabled={loading}
                  >
                    {languages.map((lang) => (
                      <MenuItem key={lang.value} value={lang.value}>
                        {lang.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </>
            )}
            <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{ mt: 3, mb: 2 }}
              disabled={loading}
            >
              {loading ? 'Processing...' : type === 'login' ? 'Sign In' : 'Register'}
            </Button>
            <Box sx={{ mt: 2 }}>
              {type === 'login' ? (
                <Button
                  color="inherit"
                  onClick={() => navigate('/register')}
                >
                  Don't have an account? Register
                </Button>
              ) : (
                <Button
                  color="inherit"
                  onClick={() => navigate('/login')}
                >
                  Already have an account? Login
                </Button>
              )}
            </Box>
          </Box>
        </Paper>
      </Box>
    </Container>
  );
};

export default AuthForm;
