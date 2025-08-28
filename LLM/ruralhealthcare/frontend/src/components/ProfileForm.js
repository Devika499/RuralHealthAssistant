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
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { useAuth } from '../hooks/useAuth';
import { useLanguage } from '../hooks/useLanguage';
import { useChat } from '../hooks/useChat';
import EditIcon from '@mui/icons-material/Edit';
import CloseIcon from '@mui/icons-material/Close';

const ProfileForm = () => {
  const { user, updateUser, logout } = useAuth();
  const { language, setLanguage } = useLanguage();
  const { messages, clearMessages } = useChat();
  const [editMode, setEditMode] = useState(false);
  const [formData, setFormData] = useState({
    name: user?.name || '',
    birthdate: user?.birthdate || '',
    preferredLanguage: user?.preferredLanguage || 'hi',
  });
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [open, setOpen] = useState(false);

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
    try {
      await updateUser(formData);
      setSuccess('Profile updated successfully');
      setEditMode(false);
    } catch (err) {
      setError(err.message || 'Failed to update profile');
    }
  };

  const handleLogout = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  const handleConfirmLogout = async () => {
    await logout(messages, clearMessages);
    handleClose();
  };

  return (
    <Container component="main" maxWidth="sm">
      <Paper elevation={3} sx={{ p: 4, mt: 8 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 4 }}>
          <Typography variant="h5">Profile</Typography>
          <Button variant="outlined" color="error" onClick={handleLogout}>
            Logout
          </Button>
        </Box>

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

        <Box component="form" onSubmit={handleSubmit}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <Avatar
              sx={{
                width: 64,
                height: 64,
                bgcolor: '#1976d2',
                mr: 2,
              }}
            >
              {user?.name ? user.name[0] : 'U'}
            </Avatar>
            <Typography variant="h6">{user?.name}</Typography>
          </Box>

          <TextField
            fullWidth
            label="Mobile Number"
            value={user?.mobileNumber}
            disabled
            sx={{ mb: 2 }}
          />

          <TextField
            fullWidth
            label="Name"
            name="name"
            value={formData.name}
            onChange={handleChange}
            disabled={!editMode}
            sx={{ mb: 2 }}
          />

          <TextField
            fullWidth
            label="Birthdate"
            type="date"
            name="birthdate"
            value={formData.birthdate}
            onChange={handleChange}
            disabled={!editMode}
            InputLabelProps={{
              shrink: true,
            }}
            sx={{ mb: 2 }}
          />

          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Preferred Language</InputLabel>
            <Select
              name="preferredLanguage"
              value={formData.preferredLanguage}
              onChange={handleChange}
              disabled={!editMode}
              label="Preferred Language"
            >
              {languages.map((lang) => (
                <MenuItem key={lang.value} value={lang.value}>
                  {lang.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <Box sx={{ mt: 3 }}>
            {!editMode ? (
              <Button
                variant="contained"
                color="primary"
                onClick={() => setEditMode(true)}
                startIcon={<EditIcon />}
              >
                Edit Profile
              </Button>
            ) : (
              <Box sx={{ display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  color="primary"
                  type="submit"
                >
                  Save Changes
                </Button>
                <Button
                  variant="outlined"
                  onClick={() => setEditMode(false)}
                >
                  Cancel
                </Button>
              </Box>
            )}
          </Box>
        </Box>
      </Paper>

      <Dialog open={open} onClose={handleClose}>
        <DialogTitle>Confirm Logout</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <Button
              variant="contained"
              color="error"
              onClick={handleConfirmLogout}
              sx={{ mr: 2 }}
            >
              Logout
            </Button>
            <Button onClick={handleClose}>Cancel</Button>
          </Box>
        </DialogContent>
      </Dialog>
    </Container>
  );
};

export default ProfileForm;
