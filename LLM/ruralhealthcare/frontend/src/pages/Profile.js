import React, { useState } from 'react';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  Alert,
  Avatar,
  Dialog,
  DialogTitle,
  DialogContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { useAuth } from '../hooks/useAuth';
import EditIcon from '@mui/icons-material/Edit';
import CloseIcon from '@mui/icons-material/Close';

const Profile = () => {
  const { user, updateUser, logout } = useAuth();
  const [editMode, setEditMode] = useState(false);
  const [formData, setFormData] = useState({
    name: user?.name || '',
    birthdate: user?.birthdate || '',
    preferredLanguage: user?.preferredLanguage || 'hi',
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
    try {
      await updateUser(formData);
      setSuccess('Profile updated successfully');
      setEditMode(false);
    } catch (err) {
      setError(err.message || 'Failed to update profile');
    }
  };

  const handleLogout = () => {
    logout();
  };

  return (
    <Container component="main" maxWidth="sm">
      <Paper elevation={3} sx={{ p: 4, mt: 8 }}>
        <Box sx={{ mb: 3 }}>
          <Typography variant="h5" component="h1" gutterBottom>
            Profile
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
        </Box>

        <Box sx={{ textAlign: 'center', mb: 3 }}>
          <Avatar
            sx={{
              width: 100,
              height: 100,
              mb: 2,
            }}
          >
            {user?.name?.[0] || ''}
          </Avatar>
          <Typography variant="h6" gutterBottom>
            {user?.name}
          </Typography>
          <Typography color="textSecondary">
            Mobile: {user?.mobileNumber}
          </Typography>
        </Box>

        {editMode ? (
          <form onSubmit={handleSubmit}>
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

            <FormControl fullWidth margin="normal">
              <InputLabel>Preferred Language</InputLabel>
              <Select
                name="preferredLanguage"
                value={formData.preferredLanguage}
                onChange={handleChange}
                label="Preferred Language"
              >
                <MenuItem value="hi">Hindi</MenuItem>
                <MenuItem value="en">English</MenuItem>
              </Select>
            </FormControl>

            <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
              <Button
                variant="contained"
                type="submit"
              >
                Save
              </Button>
              <Button
                variant="outlined"
                onClick={() => setEditMode(false)}
              >
                Cancel
              </Button>
            </Box>
          </form>
        ) : (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body1">Name: {user?.name}</Typography>
              <EditIcon
                onClick={() => setEditMode(true)}
                sx={{ cursor: 'pointer' }}
              />
            </Box>

            <Typography variant="body1">Birthdate: {user?.birthdate}</Typography>
            <Typography variant="body1">Preferred Language: {user?.preferredLanguage}</Typography>

            <Button
              variant="contained"
              color="error"
              onClick={handleLogout}
              sx={{ mt: 2 }}
            >
              Logout
            </Button>
          </Box>
        )}
      </Paper>
    </Container>
  );
};

export default Profile;
