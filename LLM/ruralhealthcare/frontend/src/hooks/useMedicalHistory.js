import { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const useMedicalHistory = () => {
  const [medicalRecords, setMedicalRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadMedicalRecords();
  }, []);

  const loadMedicalRecords = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_URL}/medical-history`);
      setMedicalRecords(response.data);
    } catch (err) {
      setError('Failed to load medical records');
    } finally {
      setLoading(false);
    }
  };

  const addRecord = async (record) => {
    try {
      await axios.post(`${API_URL}/medical-history`, record);
      loadMedicalRecords();
    } catch (err) {
      throw new Error('Failed to add medical record');
    }
  };

  const updateRecord = async (record) => {
    try {
      await axios.put(`${API_URL}/medical-history/${record.id}`, record);
      loadMedicalRecords();
    } catch (err) {
      throw new Error('Failed to update medical record');
    }
  };

  const deleteRecord = async (id) => {
    try {
      await axios.delete(`${API_URL}/medical-history/${id}`);
      loadMedicalRecords();
    } catch (err) {
      throw new Error('Failed to delete medical record');
    }
  };

  return {
    medicalRecords,
    loading,
    error,
    addRecord,
    updateRecord,
    deleteRecord,
  };
};
